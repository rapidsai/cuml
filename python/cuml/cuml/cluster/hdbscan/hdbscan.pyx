# Copyright (c) 2021-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# distutils: language = c++
import cupy as cp
import numpy as np
from pylibraft.common.handle import Handle

import cuml
from cuml.common import input_to_cuml_array
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.doc_utils import generate_docstring
from cuml.internals import logger
from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from cuml.internals.interop import (
    InteropMixin,
    UnsupportedOnGPU,
    to_cpu,
    to_gpu,
)
from cuml.internals.mixins import ClusterMixin, CMajorInputTagMixin

from cython.operator cimport dereference as deref
from libc.stdint cimport int64_t, uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t
from rmm.librmm.device_uvector cimport device_uvector

cimport cuml.cluster.hdbscan.headers as lib
from cuml.metrics.distance_type cimport DistanceType


def import_hdbscan():
    """Import `hdbscan`, raising a nicer error if missing"""
    try:
        import hdbscan
        return hdbscan
    except ImportError as exc:
        raise ImportError(
            "The `hdbscan` library is required to use this functionality, "
            "but is not currently installed."
        ) from exc


_metrics_mapping = {
    "l2": DistanceType.L2SqrtExpanded,
    "euclidean": DistanceType.L2SqrtExpanded,
}


def _cupy_array_from_ptr(ptr, shape, dtype, owner):
    """Create a cupy array from a pointer and metadata."""
    dtype = np.dtype(dtype)
    mem = cp.cuda.UnownedMemory(
        ptr=ptr, size=np.prod(shape) * dtype.itemsize, owner=owner,
    )
    mem_ptr = cp.cuda.memory.MemoryPointer(mem, 0)
    return cp.ndarray(shape=shape, dtype=dtype, memptr=mem_ptr)


cdef class _HDBSCANState:
    """Holds internal state of a fit HDBSCAN model.

    This class exists to decouple the c++ classes backing the results of an
    HDBSCAN fit from an `HDBSCAN` estimator itself. This helps ensure that
    refitting the same estimator doesn't invalidate old arrays that may still
    be lingering in memory.

    Further, it handles all the ways to coerce other values into internal state
    of `HDBSCAN`:

    - `init_and_fit`: for a direct call to `HDBSCAN.fit`
    - `from_dendrogram`: for testing comparisons to upstream `hdbscan`
    - `from_dict`: for unpickling
    - `from_sklearn`: for coercion from `hdbscan.HDBSCAN`

    The wrapper class helps isolate all `new`/`del` calls from the rest of the
    code. Any exposed output arrays have their lifetimes tied to it, ensuring
    that memory won't be freed until no python references still exist.
    """

    # A pointer to an `hdbscan_output` instance from a `fit` call, or `NULL` if
    # this state was initialized through a different method
    cdef lib.hdbscan_output *hdbscan_output

    # A pointer to a `CondensedHierarchy`, or `NULL` if this state was
    # initialized through a `fit` call.
    cdef lib.CondensedHierarchy[int64_t, float] *condensed_tree

    # The generated PredictionData, or NULL if prediction data was not yet generated.
    cdef lib.PredictionData[int64_t, float] *prediction_data

    # The number of clusters
    cdef public int n_clusters

    # A CumlArray. Either a view of `hdbscan_output`, or memory allocated by
    # `CumlArray` (if not from a `fit` call). This is also passed to
    # `PredictionData`, which keeps a view of it (but doesn't do anything to
    # keep it alive while using it). We keep a reference here to work around
    # that.
    cdef object core_dists

    # A CumlArray. Either a view of `hdbscan_output`, or memory allocated by
    # `CumlArray` (if not from a `fit` call).
    cdef object inverse_label_map

    # A cached numpy array of the condensed tree, or None if a host-array version
    # of the tree hasn't been requested yet.
    cdef object cached_condensed_tree

    def __dealloc__(self):
        if self.prediction_data != NULL:
            del self.prediction_data
            self.prediction_data = NULL
        if self.condensed_tree != NULL:
            del self.condensed_tree
            self.condensed_tree = NULL
        if self.hdbscan_output != NULL:
            del self.hdbscan_output
            self.hdbscan_output = NULL

    def to_dict(self):
        """Returns a dict that can later be passed to `from_dict` to recreate state."""
        return {
            "n_leaves": self.get_condensed_tree().get_n_leaves(),
            "n_clusters": self.n_clusters,
            "core_dists": self.core_dists,
            "inverse_label_map": self.inverse_label_map,
            "condensed_tree": self.get_condensed_tree_array(),
        }

    def _init_from_condensed_tree_array(self, handle, tree, n_leaves):
        """Shared helper for initializing a `CondensedHierarchy` from a condensed_tree array"""
        self.cached_condensed_tree = tree

        parents = np.ascontiguousarray(tree["parent"], dtype=np.int64)
        children = np.ascontiguousarray(tree["child"], dtype=np.int64)
        lambdas = np.ascontiguousarray(tree["lambda_val"], dtype=np.float32)
        sizes = np.ascontiguousarray(tree["child_size"], dtype=np.int64)

        cdef int n_edges = len(tree)
        cdef handle_t *handle_ = <handle_t*> <size_t> handle.getHandle()
        self.condensed_tree = new lib.CondensedHierarchy[int64_t, float](
            handle_[0],
            n_leaves,
            n_edges,
            <int64_t*><uintptr_t>(parents.ctypes.data),
            <int64_t*><uintptr_t>(children.ctypes.data),
            <float*><uintptr_t>(lambdas.ctypes.data),
            <int64_t*><uintptr_t>(sizes.ctypes.data),
        )

    @staticmethod
    def from_dict(handle, mapping):
        """Initialize internal state from the output of `to_dict`."""
        cdef _HDBSCANState self = _HDBSCANState.__new__(_HDBSCANState)
        self.n_clusters = mapping["n_clusters"]
        self.core_dists = mapping["core_dists"]
        self.inverse_label_map = mapping["inverse_label_map"]
        self._init_from_condensed_tree_array(
            handle, mapping["condensed_tree"], mapping["n_leaves"]
        )
        return self

    @staticmethod
    def from_sklearn(handle, model, X):
        """Initialize internal state from a `hdbscan.HDBSCAN` instance."""
        cdef DistanceType metric = _metrics_mapping[model.metric]
        cdef lib.CLUSTER_SELECTION_METHOD cluster_selection_method = {
            "eom": lib.CLUSTER_SELECTION_METHOD.EOM,
            "leaf": lib.CLUSTER_SELECTION_METHOD.LEAF,
        }[model.cluster_selection_method]

        cdef _HDBSCANState self = _HDBSCANState.__new__(_HDBSCANState)

        cdef int n_rows = X.shape[0]
        cdef int n_cols = X.shape[1]

        self._init_from_condensed_tree_array(handle, model._condensed_tree, n_rows)

        self.core_dists = CumlArray.empty(n_rows, dtype=np.float32)
        cdef handle_t *handle_ = <handle_t*> <size_t> handle.getHandle()
        cdef float* X_ptr = <float*><uintptr_t>X.ptr
        cdef float* core_dists_ptr = <float*><uintptr_t>self.core_dists.ptr
        cdef bool allow_single_cluster = model.allow_single_cluster
        cdef int64_t max_cluster_size = model.max_cluster_size
        cdef float cluster_selection_epsilon = model.cluster_selection_epsilon
        cdef int min_samples = model.min_samples or model.min_cluster_size
        cdef device_uvector[int64_t] *temp_buffer = new device_uvector[int64_t](
            0,
            handle_[0].get_stream(),
        )

        with nogil:
            lib.compute_core_dists(
                handle_[0],
                X_ptr,
                core_dists_ptr,
                n_rows,
                n_cols,
                metric,
                min_samples
            )
            lib.compute_inverse_label_map(
                handle_[0],
                deref(self.condensed_tree),
                n_rows,
                cluster_selection_method,
                deref(temp_buffer),
                allow_single_cluster,
                max_cluster_size,
                cluster_selection_epsilon
            )
        handle.sync()

        self.n_clusters = temp_buffer.size()

        if self.n_clusters > 0:
            self.inverse_label_map = CumlArray(
                data=_cupy_array_from_ptr(
                    <size_t>temp_buffer.data(),
                    (self.n_clusters,),
                    np.int64,
                    self
                ).copy()
            )
        else:
            self.inverse_label_map = CumlArray.empty((0,), dtype=np.int64)

        del temp_buffer

        return self

    cdef lib.CondensedHierarchy[int64_t, float]* get_condensed_tree(self) nogil:
        if self.hdbscan_output != NULL:
            return &(self.hdbscan_output.get_condensed_tree())
        return self.condensed_tree

    @staticmethod
    def from_dendrogram(dendrogram, int min_cluster_size):
        """Initialize internal state from a ScipPy dendrogram.

        Parameters
        ----------
        dendrogram : array-like (size n_samples, 4)
            Dendrogram in Scipy hierarchy format

        min_cluster_size : int
            Minimum number of children for a cluster to persist
        """
        cdef _HDBSCANState self = _HDBSCANState.__new__(_HDBSCANState)

        children = input_to_cuml_array(
            dendrogram[:, 0:2],
            order='C',
            check_dtype=[np.int64],
            convert_to_dtype=np.int64,
        )[0]

        lambdas = input_to_cuml_array(
            dendrogram[:, 2],
            order='C',
            check_dtype=[np.float32],
            convert_to_dtype=np.float32,
        )[0]

        sizes = input_to_cuml_array(
            dendrogram[:, 3],
            order='C',
            check_dtype=[np.int64],
            convert_to_dtype=np.int64,
        )[0]

        cdef size_t n_leaves = dendrogram.shape[0] + 1

        handle = Handle()
        cdef handle_t *handle_ = <handle_t*> <size_t> handle.getHandle()

        self.condensed_tree = new lib.CondensedHierarchy[int64_t, float](handle_[0], n_leaves)
        cdef int64_t* children_ptr = <int64_t*><uintptr_t>children.ptr
        cdef float* lambdas_ptr = <float*><uintptr_t>lambdas.ptr
        cdef int64_t* sizes_ptr = <int64_t*><uintptr_t>sizes.ptr
        with nogil:
            lib.build_condensed_hierarchy(
                handle_[0],
                children_ptr,
                lambdas_ptr,
                sizes_ptr,
                min_cluster_size,
                n_leaves,
                deref(self.condensed_tree)
            )
        return self

    @staticmethod
    cdef init_and_fit(
        handle,
        X,
        lib.HDBSCANParams params,
        DistanceType metric,
        bool gen_min_span_tree,
    ):
        """Initialize internal state from a new `fit`"""
        cdef _HDBSCANState self = _HDBSCANState.__new__(_HDBSCANState)

        cdef int n_rows = X.shape[0]
        cdef int n_cols = X.shape[1]

        # Allocate output structures
        labels = CumlArray.empty(n_rows, dtype="int64", index=X.index)
        probabilities = CumlArray.empty(n_rows, dtype="float32")

        children = CumlArray.empty((2, n_rows), dtype="int64")
        lambdas = CumlArray.empty(n_rows, dtype="float32")
        sizes = CumlArray.empty(n_rows, dtype="int64")

        mst_src = CumlArray.empty(n_rows - 1, dtype="int64")
        mst_dst = CumlArray.empty(n_rows - 1, dtype="int64")
        mst_weights = CumlArray.empty(n_rows - 1, dtype="float32")

        core_dists = CumlArray.empty(n_rows, dtype="float32")

        cdef handle_t* handle_ = <handle_t*><uintptr_t>handle.getHandle()
        cdef float* X_ptr = <float*><uintptr_t>X.ptr
        cdef float* core_dists_ptr = <float*><uintptr_t>core_dists.ptr

        self.hdbscan_output = new lib.hdbscan_output(
            handle_[0],
            n_rows,
            <int64_t*><uintptr_t>(labels.ptr),
            <float*><uintptr_t>(probabilities.ptr),
            <int64_t*><uintptr_t>(children.ptr),
            <int64_t*><uintptr_t>(sizes.ptr),
            <float*><uintptr_t>(lambdas.ptr),
            <int64_t*><uintptr_t>(mst_src.ptr),
            <int64_t*><uintptr_t>(mst_dst.ptr),
            <float*><uintptr_t>(mst_weights.ptr)
        )

        # Execute fit
        with nogil:
            lib.hdbscan(
                handle_[0],
                X_ptr,
                n_rows,
                n_cols,
                metric,
                params,
                deref(self.hdbscan_output),
                core_dists_ptr,
            )
        handle.sync()

        # Extract and store local state
        self.n_clusters = self.hdbscan_output.get_n_clusters()
        if self.n_clusters > 0:
            self.inverse_label_map = CumlArray(
                data=_cupy_array_from_ptr(
                    <size_t>self.hdbscan_output.get_inverse_label_map(),
                    (self.n_clusters,),
                    np.int64,
                    self
                )
            )
        else:
            self.inverse_label_map = CumlArray.empty((0,), dtype=np.int64)
        self.core_dists = core_dists

        # Extract and prepare results
        if self.n_clusters > 0:
            cluster_persistence = CumlArray(
                data=_cupy_array_from_ptr(
                    <size_t>self.hdbscan_output.get_stabilities(),
                    (self.n_clusters,),
                    np.float32,
                    self,
                )
            )
        else:
            cluster_persistence = CumlArray.empty((0,), dtype=np.float32)

        if gen_min_span_tree:
            min_span_tree = np.column_stack(
                (
                    mst_src.to_output("numpy", output_dtype=np.float64),
                    mst_dst.to_output("numpy", output_dtype=np.float64),
                    mst_weights.to_output("numpy", output_dtype=np.float64),
                )
            ).astype(np.float64)
        else:
            min_span_tree = None

        single_linkage_tree = np.concatenate(
            (
                children.to_output("numpy", output_dtype=np.float64),
                lambdas.to_output("numpy", output_dtype=np.float64)[None, :],
                sizes.to_output("numpy", output_dtype=np.float64)[None, :],
            ),
        )[:, :n_rows - 1].T

        return (
            self,
            self.n_clusters,
            labels,
            probabilities,
            cluster_persistence,
            min_span_tree,
            single_linkage_tree,
        )

    def generate_prediction_data(self, handle, X, labels):
        """Generate `prediction_data` if it hasn't already been generated."""
        if self.prediction_data != NULL:
            return

        cdef int n_rows = X.shape[0]
        cdef int n_cols = X.shape[1]
        cdef int64_t* labels_ptr = <int64_t*><uintptr_t>labels.ptr
        cdef int64_t* inverse_label_map_ptr = <int64_t*><uintptr_t>self.inverse_label_map.ptr
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

        self.prediction_data = new lib.PredictionData[int64_t, float](
            handle_[0],
            n_rows,
            n_cols,
            <float*><uintptr_t>(self.core_dists.ptr),
        )

        cdef lib.CondensedHierarchy[int64_t, float] *condensed_tree = self.get_condensed_tree()

        with nogil:
            lib.generate_prediction_data(
                handle_[0],
                deref(condensed_tree),
                labels_ptr,
                inverse_label_map_ptr,
                self.n_clusters,
                deref(self.prediction_data),
            )
        handle.sync()

    def get_condensed_tree_array(self):
        """Coerce `condensed_tree` to the same array layout that `hdbscan.HDBSCAN` uses."""
        if self.cached_condensed_tree is not None:
            # Cached, return the same result
            return self.cached_condensed_tree

        cdef lib.CondensedHierarchy[int64_t, float]* condensed_tree = self.get_condensed_tree()

        n_condensed_tree_edges = condensed_tree.get_n_edges()

        tree = np.recarray(
            shape=(n_condensed_tree_edges,),
            formats=[np.intp, np.intp, np.float64, np.intp],
            names=('parent', 'child', 'lambda_val', 'child_size'),
        )

        parents = _cupy_array_from_ptr(
            <size_t>condensed_tree.get_parents(),
            (n_condensed_tree_edges,),
            np.int64,
            self,
        )

        children = _cupy_array_from_ptr(
            <size_t>condensed_tree.get_children(),
            (n_condensed_tree_edges,),
            np.int64,
            self,
        )

        lambdas = _cupy_array_from_ptr(
            <size_t>condensed_tree.get_lambdas(),
            (n_condensed_tree_edges,),
            np.float32,
            self,
        )

        sizes = _cupy_array_from_ptr(
            <size_t>condensed_tree.get_sizes(),
            (n_condensed_tree_edges,),
            np.int64,
            self,
        )

        tree['parent'] = parents.get()
        tree['child'] = children.get()
        tree['lambda_val'] = lambdas.get()
        tree['child_size'] = sizes.get()

        self.cached_condensed_tree = tree

        return tree


class HDBSCAN(Base, InteropMixin, ClusterMixin, CMajorInputTagMixin):
    """
    HDBSCAN Clustering

    Recursively merges the pair of clusters that minimally increases a
    given linkage distance.

    Note that while the algorithm is generally deterministic and should
    provide matching results between RAPIDS and the Scikit-learn Contrib
    versions, the construction of the k-nearest neighbors graph and
    minimum spanning tree can introduce differences between the two
    algorithms, especially when several nearest neighbors around a
    point might have the same distance. While the differences in
    the minimum spanning trees alone might be subtle, they can
    (and often will) lead to some points being assigned different
    cluster labels between the two implementations.

    Parameters
    ----------
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.

    alpha : float, optional (default=1.0)
        A distance scaling parameter as used in robust single linkage.

    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

    min_cluster_size : int, optional (default = 5)
        The minimum number of samples in a group for that group to be
        considered a cluster; groupings smaller than this size will be left
        as noise.

    min_samples : int, optional (default=None)
        The number of samples in a neighborhood for a point
        to be considered as a core point. This includes the point itself.
        If 'None', it defaults to the min_cluster_size.

    cluster_selection_epsilon : float, optional (default=0.0)
        A distance threshold. Clusters below this value will be merged.
        Note that this should not be used
        if we want to predict the cluster labels for new points in future
        (e.g. using approximate_predict), as the approximate_predict function
        is not aware of this argument.

    max_cluster_size : int, optional (default=0)
        A limit to the size of clusters returned by the eom algorithm.
        Has no effect when using leaf clustering (where clusters are
        usually small regardless) and can also be overridden in rare
        cases by a high value for cluster_selection_epsilon. Note that
        this should not be used if we want to predict the cluster labels
        for new points in future (e.g. using approximate_predict), as
        the approximate_predict function is not aware of this argument.

    metric : string, optional (default='euclidean')
        The metric to use when calculating distance between instances in a
        feature array. Allowed values: 'euclidean'.

    p : int, optional (default=None)
        p value to use if using the minkowski metric.

    cluster_selection_method : string, optional (default='eom')
        The method used to select clusters from the condensed tree. The
        standard approach for HDBSCAN* is to use an Excess of Mass algorithm
        to find the most persistent clusters. Alternatively you can instead
        select the clusters at the leaves of the tree -- this provides the
        most fine grained and homogeneous clusters. Options are:

            * ``eom``
            * ``leaf``

    allow_single_cluster : bool, optional (default=False)
        By default HDBSCAN* will not produce a single cluster, setting this
        to True will override this and allow single cluster results in
        the case that you feel this is a valid result for your dataset.

    gen_min_span_tree : bool, optional (default=False)
        Whether to populate the `minimum_spanning_tree_` member for
        utilizing plotting tools. This requires the `hdbscan` CPU Python
        package to be installed.

    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    prediction_data : bool, optional (default=False)
        Whether to generate extra cached data for predicting labels or
        membership vectors few new unseen points later. If you wish to
        persist the clustering object for later re-use you probably want
        to set this to True.

    Attributes
    ----------
    labels_ : ndarray, shape (n_samples, )
        Cluster labels for each point in the dataset given to fit().
        Noisy samples are given the label -1.

    probabilities_ : ndarray, shape (n_samples, )
        The strength with which each sample is a member of its assigned
        cluster. Noise points have probability zero; points in clusters
        have values assigned proportional to the degree that they
        persist as part of the cluster.

    cluster_persistence_ : ndarray, shape (n_clusters, )
        A score of how persistent each cluster is. A score of 1.0 represents
        a perfectly stable cluster that persists over all distance scales,
        while a score of 0.0 represents a perfectly ephemeral cluster. These
        scores can be used to gauge the relative coherence of the
        clusters output by the algorithm.

    condensed_tree_ : CondensedTree object
        The condensed tree produced by HDBSCAN. The object has methods
        for converting to pandas, networkx, and plotting.

    single_linkage_tree_ : SingleLinkageTree object
        The single linkage tree produced by HDBSCAN. The object has methods
        for converting to pandas, networkx, and plotting.

    minimum_spanning_tree_ : MinimumSpanningTree object
        The minimum spanning tree of the mutual reachability graph generated
        by HDBSCAN. Note that this is not generated by default and will only
        be available if `gen_min_span_tree` was set to True on object creation.
        Even then in some optimized cases a tree may not be generated.

    """
    labels_ = CumlArrayDescriptor()
    probabilities_ = CumlArrayDescriptor()
    cluster_persistence_ = CumlArrayDescriptor()

    _cpu_class_path = "hdbscan.HDBSCAN"

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "min_cluster_size",
            "min_samples",
            "cluster_selection_epsilon",
            "max_cluster_size",
            "metric",
            "alpha",
            "p",
            "cluster_selection_method",
            "allow_single_cluster",
            "gen_min_span_tree",
            "prediction_data"
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        if callable(model.metric) or model.metric not in _metrics_mapping:
            raise UnsupportedOnGPU(f"`metric={model.metric!r}` is not supported")

        if (
            isinstance(model.memory, str)
            or getattr(model.memory, "location", None) is not None
        ):
            # Result caching is unsupported
            raise UnsupportedOnGPU(f"`memory={model.memory!r}` is not supported")

        if model.match_reference_implementation:
            raise UnsupportedOnGPU("`match_reference_implementation=True` is not supported")

        if model.branch_detection_data:
            raise UnsupportedOnGPU("`branch_detection_data=True` is not supported")

        return {
            "min_cluster_size": model.min_cluster_size,
            "min_samples": model.min_samples,
            "cluster_selection_epsilon": model.cluster_selection_epsilon,
            "max_cluster_size": model.max_cluster_size,
            "metric": model.metric,
            "alpha": model.alpha,
            "p": model.p,
            "cluster_selection_method": model.cluster_selection_method,
            "allow_single_cluster": model.allow_single_cluster,
            "gen_min_span_tree": model.gen_min_span_tree,
            "prediction_data": model.prediction_data,
        }

    def _params_to_cpu(self):
        return {
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "cluster_selection_epsilon": self.cluster_selection_epsilon,
            "max_cluster_size": self.max_cluster_size,
            "metric": self.metric,
            "alpha": self.alpha,
            "p": self.p,
            "cluster_selection_method": self.cluster_selection_method,
            "allow_single_cluster": self.allow_single_cluster,
            "gen_min_span_tree": self.gen_min_span_tree,
            "prediction_data": self.prediction_data,
        }

    def _attrs_from_cpu(self, model):
        if (raw_data_cpu := getattr(model, "_raw_data", None)) is None:
            # Fit with precomputed metric
            raise UnsupportedOnGPU("Models fit with a precomputed metric are not supported")

        if not isinstance(raw_data_cpu, np.ndarray):
            # Sparse input
            raise UnsupportedOnGPU("Sparse inputs are not supported")

        raw_data = to_gpu(raw_data_cpu, order="C", dtype="float32")
        labels = to_gpu(model.labels_, order="C", dtype="int64")
        state = _HDBSCANState.from_sklearn(self.handle, model, raw_data)
        if model._prediction_data is not None:
            state.generate_prediction_data(self.handle, raw_data, labels)

        return {
            # XXX: `hdbscan.HDBSCAN` doesn't set `n_features_in_` currently, we need
            # to infer this ourselves from the raw data.
            "n_features_in_": raw_data_cpu.shape[1],
            "labels_": labels,
            "probabilities_": to_gpu(model.probabilities_, dtype="float32"),
            "cluster_persistence_": to_gpu(model.cluster_persistence_, dtype="float32"),
            "_raw_data": raw_data,
            "_raw_data_cpu": raw_data_cpu,
            "_state": state,
            "n_clusters_": state.n_clusters,
            "_single_linkage_tree": model._single_linkage_tree,
            "_min_spanning_tree": model._min_spanning_tree,
            "_prediction_data": model._prediction_data,
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        out = {
            "labels_": to_cpu(self.labels_),
            "probabilities_": to_cpu(self.probabilities_),
            "cluster_persistence_": to_cpu(self.cluster_persistence_),
            "_condensed_tree": self._condensed_tree,
            "_single_linkage_tree": self._single_linkage_tree,
            "_min_spanning_tree": self._min_spanning_tree,
            "_raw_data": self._get_raw_data_cpu(),
            "_all_finite": True,
            **super()._attrs_to_cpu(model),
        }
        if self.prediction_data:
            out["_prediction_data"] = self.prediction_data_
        return out

    def __init__(self, *,
                 min_cluster_size=5,
                 min_samples=None,
                 cluster_selection_epsilon=0.0,
                 max_cluster_size=0,
                 metric='euclidean',
                 alpha=1.0,
                 p=None,
                 cluster_selection_method='eom',
                 allow_single_cluster=False,
                 gen_min_span_tree=False,
                 handle=None,
                 verbose=False,
                 output_type=None,
                 prediction_data=False):

        super().__init__(handle=handle, verbose=verbose, output_type=output_type)
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.max_cluster_size = max_cluster_size
        self.metric = metric
        self.p = p
        self.alpha = alpha
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.gen_min_span_tree = gen_min_span_tree
        self.prediction_data = prediction_data

        self._single_linkage_tree = None
        self._min_spanning_tree = None
        self._prediction_data = None
        self._raw_data = None
        self._raw_data_cpu = None

    def _get_raw_data_cpu(self):
        if getattr(self, "_raw_data_cpu") is None:
            self._raw_data_cpu = self._raw_data.to_output("numpy")
        return self._raw_data_cpu

    @property
    def _condensed_tree(self):
        return self._state.get_condensed_tree_array()

    @property
    def condensed_tree_(self):
        if self._state is not None:
            hdbscan = import_hdbscan()
            return hdbscan.plots.CondensedTree(
                self._condensed_tree,
                self.cluster_selection_method,
                self.allow_single_cluster
            )
        raise AttributeError("No condensed tree was generated; try running fit first.")

    @property
    def minimum_spanning_tree_(self):
        if self._min_spanning_tree is not None:
            hdbscan = import_hdbscan()
            return hdbscan.plots.MinimumSpanningTree(
                self._min_spanning_tree,
                self._get_raw_data_cpu(),
            )
        raise AttributeError(
            "No minimum spanning tree was generated. Set `gen_min_span_tree=True` and refit."
        )

    @property
    def single_linkage_tree_(self):
        if self._single_linkage_tree is not None:
            hdbscan = import_hdbscan()
            return hdbscan.plots.SingleLinkageTree(self._single_linkage_tree)
        raise AttributeError("No single linkage tree was generated; try running fit first.")

    @property
    def prediction_data_(self):
        if not self.prediction_data:
            raise AttributeError(
               "Prediction data not yet generated, please call "
               "`model.generate_prediction_data()`"
            )

        if self._prediction_data is None:
            hdbscan = import_hdbscan()
            self._prediction_data = hdbscan.prediction.PredictionData(
                self._get_raw_data_cpu(),
                self.condensed_tree_,
                self.min_samples or self.min_cluster_size,
                tree_type="kdtree",
                metric=self.metric
            )

        return self._prediction_data

    def generate_prediction_data(self):
        """
        Create data that caches intermediate results used for predicting
        the label of new/unseen points. This data is only useful if you
        are intending to use functions from hdbscan.prediction.
        """
        if getattr(self, "labels_", None) is None:
            raise ValueError("The model is not trained yet (call fit() first).")

        with cuml.using_output_type("cuml"):
            labels = self.labels_

        self._state.generate_prediction_data(self.handle, self._raw_data, labels)
        self.prediction_data = True

    @generate_docstring()
    def fit(self, X, y=None, *, convert_dtype=True) -> "HDBSCAN":
        """
        Fit HDBSCAN model from features.
        """

        self._raw_data = input_to_cuml_array(
            X,
            order='C',
            check_dtype=[np.float32],
            convert_to_dtype=np.float32 if convert_dtype else None,
        )[0]
        self._raw_data_cpu = None

        # Validate and prepare hyperparameters
        if (min_samples := self.min_samples) is None:
            min_samples = self.min_cluster_size
        if not (1 <= min_samples <= 1023):
            raise ValueError(
                f"HDBSCAN requires `1 <= min_samples <= 1023`, got `{min_samples=}`"
            )

        cdef lib.HDBSCANParams params
        params.min_samples = min_samples
        params.alpha = self.alpha
        params.min_cluster_size = self.min_cluster_size
        params.max_cluster_size = self.max_cluster_size
        params.cluster_selection_epsilon = self.cluster_selection_epsilon
        params.allow_single_cluster = self.allow_single_cluster

        if self.cluster_selection_method == 'eom':
            params.cluster_selection_method = lib.CLUSTER_SELECTION_METHOD.EOM
        elif self.cluster_selection_method == 'leaf':
            params.cluster_selection_method = lib.CLUSTER_SELECTION_METHOD.LEAF
        else:
            raise ValueError(
                "`cluster_selection_method` must be one of {'eom', 'leaf'}, "
                f"got {self.cluster_selection_method!r}"
            )

        cdef DistanceType metric
        if self.metric in _metrics_mapping:
            metric = _metrics_mapping[self.metric]
        else:
            raise ValueError(
                f"metric must be one of {sorted(_metrics_mapping)}, got {self.metric!r}"
            )

        # Execute fit
        (
            state,
            n_clusters,
            labels,
            probabilities,
            cluster_persistence,
            min_spanning_tree,
            single_linkage_tree,
        ) = _HDBSCANState.init_and_fit(
            self.handle,
            self._raw_data,
            params,
            metric,
            self.gen_min_span_tree
        )

        # Store state on model
        self._state = state
        self.n_clusters_ = n_clusters
        self.labels_ = labels
        self.probabilities_ = probabilities
        self.cluster_persistence_ = cluster_persistence
        self._min_spanning_tree = min_spanning_tree
        self._single_linkage_tree = single_linkage_tree

        if self.prediction_data:
            self.generate_prediction_data()

        return self

    @generate_docstring(return_values={'name': 'preds',
                                       'type': 'dense',
                                       'description': 'Cluster indexes',
                                       'shape': '(n_samples, 1)'})
    def fit_predict(self, X, y=None) -> CumlArray:
        """
        Fit the HDBSCAN model from features and return
        cluster labels.
        """
        return self.fit(X).labels_

    def __getstate__(self):
        out = self.__dict__.copy()
        out["_raw_data_cpu"] = None
        if (state := out.pop("_state", None)) is not None:
            out["_state_dict"] = state.to_dict()
        return out

    def __setstate__(self, state):
        state_dict = state.pop("_state_dict", None)
        self.__dict__.update(state)
        if state_dict is not None:
            self._state = _HDBSCANState.from_dict(self.handle, state_dict)
        if self.prediction_data:
            self.generate_prediction_data()


###########################################################
#                  Prediction Functions                   #
###########################################################


def _check_clusterer(clusterer):
    """Validate an HDBSCAN instance is fit and has prediction data"""
    if not isinstance(clusterer, HDBSCAN):
        raise TypeError(
            f"Expected an instance of `HDBSCAN`, got {type(clusterer).__name__}"
        )

    if getattr(clusterer, "labels_", None) is None:
        raise ValueError(
            "The clusterer is not fit, please call `clusterer.fit` first"
        )
    cdef _HDBSCANState state = <_HDBSCANState?>clusterer._state

    if state.prediction_data == NULL:
        raise ValueError(
            "Prediction data not yet generated, please call "
            "`clusterer.generate_prediction_data()`"
        )

    return state


@cuml.internals.api_return_array()
def all_points_membership_vectors(clusterer, int batch_size=4096):
    """
    Predict soft cluster membership vectors for all points in the
    original dataset the clusterer was trained on. This function is more
    efficient by making use of the fact that all points are already in the
    condensed tree, and processing in bulk.

    Parameters
    ----------
    clusterer : HDBSCAN
        A clustering object that has been fit to the data and
        had ``prediction_data=True`` set.

    batch_size : int, optional, default=min(4096, n_rows)
        Lowers memory requirement by computing distance-based membership
        in smaller batches of points in the training data. For example, a batch
        size of 1,000 computes distance based memberships for 1,000 points at a
        time. The default batch size is 4,096.

    Returns
    -------
    membership_vectors : array (n_samples, n_clusters)
        The probability that point ``i`` of the original dataset is a member of
        cluster ``j`` is in ``membership_vectors[i, j]``.
    """
    _check_clusterer(clusterer)

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    # Reflect the output type from global settings or the clusterer
    cuml.internals.set_api_output_type(clusterer._get_output_type())

    n_rows = clusterer._raw_data.shape[0]

    if clusterer.n_clusters_ == 0:
        return CumlArray.zeros(
            n_rows,
            dtype=np.float32,
            order="C",
            index=clusterer._raw_data.index,
        )

    membership_vec = CumlArray.empty(
        (n_rows, clusterer.n_clusters_,),
        dtype="float32",
        order="C",
        index=clusterer._raw_data.index,
    )

    cdef _HDBSCANState state = <_HDBSCANState?>clusterer._state
    cdef float* X_ptr = <float*><uintptr_t>clusterer._raw_data.ptr
    cdef float* membership_vec_ptr = <float*><uintptr_t>membership_vec.ptr
    cdef DistanceType metric = _metrics_mapping[clusterer.metric]
    cdef handle_t* handle_ = <handle_t*><size_t>clusterer.handle.getHandle()

    with nogil:
        lib.compute_all_points_membership_vectors(
            handle_[0],
            deref(state.get_condensed_tree()),
            deref(state.prediction_data),
            X_ptr,
            metric,
            membership_vec_ptr,
            batch_size
        )
    clusterer.handle.sync()

    return membership_vec


@cuml.internals.api_return_array()
def membership_vector(clusterer, points_to_predict, int batch_size=4096, convert_dtype=True):
    """
    Predict soft cluster membership. The result produces a vector
    for each point in ``points_to_predict`` that gives a probability that
    the given point is a member of a cluster for each of the selected clusters
    of the ``clusterer``.

    Parameters
    ----------
    clusterer : HDBSCAN
        A clustering object that has been fit to the data and
        either had ``prediction_data=True`` set, or called the
        ``generate_prediction_data`` method after the fact.

    points_to_predict : array, or array-like (n_samples, n_features)
        The new data points to predict cluster labels for. They should
        have the same dimensionality as the original dataset over which
        clusterer was fit.

    batch_size : int, optional, default=min(4096, n_points_to_predict)
        Lowers memory requirement by computing distance-based membership
        in smaller batches of points in the prediction data. For example, a
        batch size of 1,000 computes distance based memberships for 1,000
        points at a time. The default batch size is 4,096.

    Returns
    -------
    membership_vectors : array (n_samples, n_clusters)
        The probability that point ``i`` is a member of cluster ``j`` is
        in ``membership_vectors[i, j]``.
    """
    _check_clusterer(clusterer)

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    # Reflect the output type from global settings, the clusterer, or the input
    cuml.internals.set_api_output_type(clusterer._get_output_type(points_to_predict))

    cdef int n_prediction_points
    points_to_predict_m, n_prediction_points, n_cols, _ = input_to_cuml_array(
        points_to_predict,
        order="C",
        check_dtype=[np.float32],
        convert_to_dtype=(np.float32 if convert_dtype else None)
    )

    if n_cols != clusterer._raw_data.shape[1]:
        raise ValueError("New points dimension does not match fit data!")

    if clusterer.n_clusters_ == 0:
        return CumlArray.zeros(
            n_prediction_points, dtype=np.float32, index=points_to_predict_m.index
        )

    membership_vec = CumlArray.empty(
        (n_prediction_points, clusterer.n_clusters_,),
        dtype="float32",
        order="C",
        index=points_to_predict_m.index,
    )

    cdef _HDBSCANState state = <_HDBSCANState?>clusterer._state
    cdef float* X_ptr = <float*><uintptr_t>clusterer._raw_data.ptr
    cdef float* points_to_predict_ptr = <float*><uintptr_t>points_to_predict_m.ptr
    cdef float* membership_vec_ptr = <float*><uintptr_t>membership_vec.ptr
    cdef int min_samples = clusterer.min_samples or clusterer.min_cluster_size
    cdef DistanceType metric = _metrics_mapping[clusterer.metric]
    cdef handle_t* handle_ = <handle_t*><size_t>clusterer.handle.getHandle()

    with nogil:
        lib.compute_membership_vector(
            handle_[0],
            deref(state.get_condensed_tree()),
            deref(state.prediction_data),
            X_ptr,
            points_to_predict_ptr,
            n_prediction_points,
            min_samples,
            metric,
            membership_vec_ptr,
            batch_size
        )
    clusterer.handle.sync()

    return membership_vec


@cuml.internals.api_return_generic()
def approximate_predict(clusterer, points_to_predict, convert_dtype=True):
    """Predict the cluster label of new points. The returned labels
    will be those of the original clustering found by ``clusterer``,
    and therefore are not (necessarily) the cluster labels that would
    be found by clustering the original data combined with
    ``points_to_predict``, hence the 'approximate' label.

    If you simply wish to assign new points to an existing clustering
    in the 'best' way possible, this is the function to use. If you
    want to predict how ``points_to_predict`` would cluster with
    the original data under HDBSCAN the most efficient existing approach
    is to simply recluster with the new point(s) added to the original dataset.

    Parameters
    ----------
    clusterer : HDBSCAN
        A clustering object that has been fit to the data and
        had ``prediction_data=True`` set.

    points_to_predict : array, or array-like (n_samples, n_features)
        The new data points to predict cluster labels for. They should
        have the same dimensionality as the original dataset over which
        clusterer was fit.

    Returns
    -------
    labels : array (n_samples,)
        The predicted labels of the ``points_to_predict``

    probabilities : array (n_samples,)
        The soft cluster scores for each of the ``points_to_predict``
    """
    _check_clusterer(clusterer)

    # Reflect the output type from global settings, the clusterer, or the input
    cuml.internals.set_api_output_type(clusterer._get_output_type(points_to_predict))

    if clusterer.n_clusters_ == 0:
        logger.warn(
            "Clusterer does not have any defined clusters, new data "
            "will be automatically predicted as outliers."
        )

    cdef int n_prediction_points
    points_to_predict_m, n_prediction_points, n_cols, _ = input_to_cuml_array(
        points_to_predict,
        order="C",
        check_dtype=[np.float32],
        convert_to_dtype=(np.float32 if convert_dtype else None),
    )

    if n_cols != clusterer._raw_data.shape[1]:
        raise ValueError("New points dimension does not match fit data!")

    prediction_labels = CumlArray.empty(
        (n_prediction_points,),
        dtype="int64",
        index=points_to_predict_m.index,
    )
    prediction_probs = CumlArray.empty(
        (n_prediction_points,),
        dtype="float32",
        index=points_to_predict_m.index,
    )

    with cuml.using_output_type("cuml"):
        labels = clusterer.labels_

    cdef _HDBSCANState state = <_HDBSCANState?>clusterer._state
    cdef float* X_ptr = <float*><uintptr_t>clusterer._raw_data.ptr
    cdef int64_t* labels_ptr = <int64_t*><uintptr_t>labels.ptr
    cdef float* points_to_predict_ptr = <float*><uintptr_t>points_to_predict_m.ptr
    cdef int64_t* prediction_labels_ptr = <int64_t*><uintptr_t>prediction_labels.ptr
    cdef float* prediction_probs_ptr = <float*><uintptr_t>prediction_probs.ptr
    cdef DistanceType metric = _metrics_mapping[clusterer.metric]
    cdef int min_samples = clusterer.min_samples or clusterer.min_cluster_size,
    cdef handle_t* handle_ = <handle_t*><size_t>clusterer.handle.getHandle()

    with nogil:
        lib.out_of_sample_predict(
            handle_[0],
            deref(state.get_condensed_tree()),
            deref(state.prediction_data),
            X_ptr,
            labels_ptr,
            points_to_predict_ptr,
            n_prediction_points,
            metric,
            min_samples,
            prediction_labels_ptr,
            prediction_probs_ptr,
        )
    clusterer.handle.sync()

    return prediction_labels, prediction_probs


###########################################################
#           Functions exposed for testing only            #
###########################################################

def _condense_hierarchy(dendrogram, min_cluster_size):
    """
    Accepts a dendrogram in the Scipy hierarchy format, condenses the
    dendrogram to collapse subtrees containing less than min_cluster_size
    leaves, and returns a ``condensed_tree``.

    Exposed for testing only.

    Parameters
    ----------
    dendrogram : array-like (n_samples, 4)
        Dendrogram in Scipy hierarchy format.

    min_cluster_size : int
        Minimum number of children for a cluster to persist.

    Returns
    -------
    condensed_tree : np.ndarray
    """
    state = _HDBSCANState.from_dendrogram(dendrogram, min_cluster_size)
    return state.get_condensed_tree_array()


def _extract_clusters(
    condensed_tree,
    handle=None,
    allow_single_cluster=False,
    max_cluster_size=0,
    cluster_selection_method="eom",
    cluster_selection_epsilon=0.0,
):
    """Extract clusters from a condensed_tree.

    Exposed for testing only"""
    cdef size_t n_leaves = condensed_tree['parent'].min()
    cdef int n_edges = len(condensed_tree)

    parents = input_to_cuml_array(
        condensed_tree['parent'],
        order='C',
        convert_to_dtype=np.int64,
    )[0]

    children = input_to_cuml_array(
        condensed_tree["child"],
        order='C',
        convert_to_dtype=np.int64,
    )[0]

    lambdas = input_to_cuml_array(
        condensed_tree['lambda_val'],
        order='C',
        convert_to_dtype=np.float32,
    )[0]

    sizes = input_to_cuml_array(
        condensed_tree['child_size'],
        order='C',
        convert_to_dtype=np.int64,
    )[0]

    labels = CumlArray.empty(n_leaves, dtype="int64")
    probabilities = CumlArray.empty(n_leaves, dtype="float32")

    cdef lib.CLUSTER_SELECTION_METHOD cluster_selection_method_val = {
        "eom": lib.CLUSTER_SELECTION_METHOD.EOM,
        "leaf": lib.CLUSTER_SELECTION_METHOD.LEAF,
    }[cluster_selection_method]

    if handle is None:
        handle = Handle()
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

    lib._extract_clusters(
        handle_[0],
        n_leaves,
        n_edges,
        <int64_t*><uintptr_t>(parents.ptr),
        <int64_t*><uintptr_t>(children.ptr),
        <float*><uintptr_t>(lambdas.ptr),
        <int64_t*><uintptr_t>(sizes.ptr),
        <int64_t*><uintptr_t>(labels.ptr),
        <float*><uintptr_t>(probabilities.ptr),
        cluster_selection_method_val,
        <bool> allow_single_cluster,
        <int64_t> max_cluster_size,
        <float> cluster_selection_epsilon,
    )
    handle.sync()

    return (
        labels.to_output("numpy"),
        probabilities.to_output("numpy"),
    )

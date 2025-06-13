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

from functools import cached_property

import cupy as cp
import numpy as np
from pylibraft.common.handle import Handle

import cuml
import cuml.accel
from cuml.common import input_to_cuml_array
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.doc_utils import generate_docstring
from cuml.internals.api_decorators import (
    device_interop_preparation,
    enable_device_interop,
)
from cuml.internals.array import CumlArray
from cuml.internals.base import UniversalBase
from cuml.internals.mixins import ClusterMixin, CMajorInputTagMixin

from cython.operator cimport dereference as deref
from libc.stdint cimport uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t

cimport cuml.cluster.hdbscan.headers as lib
from cuml.cluster.hdbscan.headers cimport (
    CLUSTER_SELECTION_METHOD,
    CondensedHierarchy,
    HDBSCANParams,
    PredictionData,
    hdbscan_output,
)
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
    'l2': DistanceType.L2SqrtExpanded,
    'euclidean': DistanceType.L2SqrtExpanded,
}


def _cuml_array_from_ptr(ptr, buf_size, shape, dtype, owner):
    mem = cp.cuda.UnownedMemory(ptr=ptr, size=buf_size,
                                owner=owner,
                                device_id=-1)
    mem_ptr = cp.cuda.memory.MemoryPointer(mem, 0)
    return CumlArray(data=cp.ndarray(shape=shape, dtype=dtype, memptr=mem_ptr))


def _construct_condensed_tree_attribute(ptr,
                                        n_condensed_tree_edges,
                                        dtype="int32",
                                        owner=None):

    return _cuml_array_from_ptr(
        ptr, n_condensed_tree_edges * sizeof(float),
        (n_condensed_tree_edges,), dtype, owner
    )


def _build_condensed_tree_plot_host(
        parent, child, lambdas, sizes,
        cluster_selection_method, allow_single_cluster):
    raw_tree = np.recarray(shape=(parent.shape[0],),
                           formats=[np.intp, np.intp, float, np.intp],
                           names=('parent', 'child', 'lambda_val',
                                  'child_size'))
    raw_tree['parent'] = parent
    raw_tree['child'] = child
    raw_tree['lambda_val'] = lambdas
    raw_tree['child_size'] = sizes

    hdbscan = import_hdbscan()
    return hdbscan.plots.CondensedTree(
        raw_tree, cluster_selection_method, allow_single_cluster
    )


def condense_hierarchy(dendrogram,
                       min_cluster_size,
                       allow_single_cluster=False,
                       cluster_selection_epsilon=0.0):
    """
    Accepts a dendrogram in the Scipy hierarchy format, condenses the
    dendrogram to collapse subtrees containing less than min_cluster_size
    leaves, and returns an hdbscan.plots.CondensedTree object with
    the result on host.

    Parameters
    ----------

    dendrogram : array-like (size n_samples, 4)
        Dendrogram in Scipy hierarchy format

    min_cluster_size : int minimum number of children for a cluster
        to persist

    allow_single_cluster : bool whether or not to allow a single
        cluster in the face of mostly noise.

    cluster_selection_epsilon : float minimum distance threshold used to
        determine when clusters should be merged.

    Returns
    -------

    condensed_tree : hdbscan.plots.CondensedTree object
    """

    _children, _, _, _ = \
        input_to_cuml_array(dendrogram[:, 0:2].astype('int32'), order='C',
                            check_dtype=[np.int32],
                            convert_to_dtype=(np.int32))

    _lambdas, _, _, _ = \
        input_to_cuml_array(dendrogram[:, 2], order='C',
                            check_dtype=[np.float32],
                            convert_to_dtype=(np.float32))

    _sizes, _, _, _ = \
        input_to_cuml_array(dendrogram[:, 3], order='C',
                            check_dtype=[np.int32],
                            convert_to_dtype=(np.int32))

    handle = Handle()
    cdef handle_t *handle_ = <handle_t*> <size_t> handle.getHandle()
    n_leaves = dendrogram.shape[0]+1
    cdef CondensedHierarchy[int, float] *condensed_tree =\
        new CondensedHierarchy[int, float](
            handle_[0], <size_t>n_leaves)

    cdef uintptr_t _children_ptr = _children.ptr
    cdef uintptr_t _lambdas_ptr = _lambdas.ptr
    cdef uintptr_t _sizes_ptr = _sizes.ptr

    lib.build_condensed_hierarchy(
        handle_[0],
        <int*> _children_ptr,
        <float*> _lambdas_ptr,
        <int*> _sizes_ptr,
        <int>min_cluster_size,
        n_leaves,
        deref(condensed_tree)
    )

    n_condensed_tree_edges = condensed_tree.get_n_edges()

    condensed_parent_ = _construct_condensed_tree_attribute(
        <size_t>condensed_tree.get_parents(), n_condensed_tree_edges)

    condensed_child_ = _construct_condensed_tree_attribute(
        <size_t>condensed_tree.get_children(), n_condensed_tree_edges)

    condensed_lambdas_ = \
        _construct_condensed_tree_attribute(
            <size_t>condensed_tree.get_lambdas(), n_condensed_tree_edges,
            "float32")

    condensed_sizes_ = _construct_condensed_tree_attribute(
        <size_t>condensed_tree.get_sizes(), n_condensed_tree_edges)

    condensed_tree_host = _build_condensed_tree_plot_host(
        condensed_parent_.to_output('numpy'),
        condensed_child_.to_output("numpy"),
        condensed_lambdas_.to_output("numpy"),
        condensed_sizes_.to_output("numpy"), cluster_selection_epsilon,
        allow_single_cluster)

    del condensed_tree

    return condensed_tree_host


cdef class _CondensedHierarchy:
    cdef lib.CondensedHierarchy[int, float] *ptr

    def __init__(self):
        raise TypeError("Cannot construct via `__init__`")

    @staticmethod
    cdef _CondensedHierarchy from_components(
        handle,
        n_rows,
        parents,
        children,
        lambdas,
        sizes,
    ):
        cdef _CondensedHierarchy self = _CondensedHierarchy.__new__(_CondensedHierarchy)
        cdef handle_t* handle_ = <handle_t*><uintptr_t>handle.getHandle()
        parents_m = input_to_cuml_array(parents, order="C", convert_to_dtype=np.int32)[0]
        children_m = input_to_cuml_array(children, order="C", convert_to_dtype=np.int32)[0]
        lambdas_m = input_to_cuml_array(lambdas, order="C", convert_to_dtype=np.float32)[0]
        sizes_m = input_to_cuml_array(sizes, order="C", convert_to_dtype=np.int32)[0]

        self.ptr = new lib.CondensedHierarchy[int, float](
            handle_[0],
            <size_t>n_rows,
            <int>parents_m.shape[0],
            <int*><uintptr_t>(parents_m.ptr),
            <int*><uintptr_t>(children_m.ptr),
            <float*><uintptr_t>(lambdas_m.ptr),
            <int*><uintptr_t>(sizes_m.ptr),
        )
        return self

    def __dealloc__(self):
        if self.ptr is not NULL:
            del self.ptr
            self.ptr = NULL


cdef class _PredictionData:
    cdef lib.PredictionData[int, float] *ptr

    def __init__(self, handle, n_rows, n_cols, core_dists):
        cdef handle_t* handle_ = <handle_t*><uintptr_t>handle.getHandle()

        self.ptr = new lib.PredictionData(
            handle_[0],
            <int> n_rows,
            <int> n_cols,
            <float*><uintptr_t>(core_dists.ptr),
        )

    def __dealloc__(self):
        if self.ptr != NULL:
            del self.ptr
            self.ptr = NULL


cdef class _HDBSCANOutput:
    cdef lib.hdbscan_output *ptr

    def __init__(self, handle: Handle, X_m: CumlArray):
        n_rows = len(X_m)

        cdef handle_t* handle_ = <handle_t*><uintptr_t>handle.getHandle()

        self.labels = CumlArray.empty(n_rows, dtype="int32", index=X_m.index)
        self.children = CumlArray.empty((2, n_rows), dtype="int32")
        self.probabilities = CumlArray.empty(n_rows, dtype="float32")
        self.sizes = CumlArray.empty(n_rows, dtype="int32")
        self.lambdas = CumlArray.empty(n_rows, dtype="float32")
        self.mst_src = CumlArray.empty(n_rows - 1, dtype="int32")
        self.mst_dst = CumlArray.empty(n_rows - 1, dtype="int32")
        self.mst_weights = CumlArray.empty(n_rows - 1, dtype="float32")

        cdef uintptr_t labels_ptr = self.labels.ptr
        cdef uintptr_t children_ptr = self.children.ptr
        cdef uintptr_t sizes_ptr = self.sizes.ptr
        cdef uintptr_t lambdas_ptr = self.lambdas.ptr
        cdef uintptr_t probabilities_ptr = self.probabilities.ptr
        cdef uintptr_t mst_src_ptr = self.mst_src.ptr
        cdef uintptr_t mst_dst_ptr = self.mst_dst.ptr
        cdef uintptr_t mst_weights_ptr = self.mst_weights.ptr

        self.ptr = new lib.hdbscan_output(
            handle_[0],
            n_rows,
            <int*>labels_ptr,
            <float*>probabilities_ptr,
            <int*>children_ptr,
            <int*>sizes_ptr,
            <float*>lambdas_ptr,
            <int*>mst_src_ptr,
            <int*>mst_dst_ptr,
            <float*>mst_weights_ptr
        )

    def __dealloc__(self):
        if self.ptr is not NULL:
            del self.ptr
            self.ptr = NULL

    def get_n_clusters(self):
        return self.ptr.get_n_clusters()

    def get_cluster_persistence(self):
        n_clusters = self.ptr.get_n_clusters()
        if n_clusters > 0:
            return _cuml_array_from_ptr(
                <size_t>self.ptr.get_stabilities(),
                n_clusters * sizeof(float),
                (1, n_clusters),
                "float32",
                self,
            )
        else:
            return CumlArray.empty((0,), dtype="float32")

    def get_inverse_label_map(self):
        n_clusters = self.ptr.get_n_clusters()
        if n_clusters > 0:
            return _cuml_array_from_ptr(
                <size_t>self.ptr.get_inverse_label_map(),
                n_clusters * sizeof(int),
                (self.n_clusters_, ),
                "int32",
                self
            )
        else:
            return CumlArray.empty((0,), dtype="int32")

    def get_condensed_tree_components(self):
        n_condensed_tree_edges = self.ptr.get_condensed_tree().get_n_edges()

        parents = _cuml_array_from_ptr(
            <size_t>self.ptr.get_condensed_tree().get_parents(),
            n_condensed_tree_edges * sizeof(int),
            (n_condensed_tree_edges,),
            "int32",
            self,
        )

        children = _cuml_array_from_ptr(
            <size_t>self.ptr.get_condensed_tree().get_children(),
            n_condensed_tree_edges * sizeof(int),
            (n_condensed_tree_edges,),
            "int32",
            self,
        )

        lambdas = _cuml_array_from_ptr(
            <size_t>self.ptr.get_condensed_tree().get_lambdas(),
            n_condensed_tree_edges * sizeof(float),
            (n_condensed_tree_edges,),
            "float32",
            self,
        )

        sizes = _cuml_array_from_ptr(
            <size_t>self.ptr.get_condensed_tree().get_sizes(),
            n_condensed_tree_edges * sizeof(int),
            (n_condensed_tree_edges,),
            "int32",
            self,
        )
        return parents, children, lambdas, sizes


class HDBSCAN(UniversalBase, ClusterMixin, CMajorInputTagMixin):

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

    p : int, optional (default=2)
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

    gen_condensed_tree : bool, optional (default=False)
        Whether to populate the `condensed_tree_` member for
        utilizing plotting tools. This requires the `hdbscan` CPU
        Python package to be installed.

    gen_single_linkage_tree_ : bool, optional (default=False)
        Whether to populate the `single_linkage_tree_` member for
        utilizing plotting tools. This requires the `hdbscan` CPU
        Python package t be installed.

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
    _cpu_estimator_import_path = 'hdbscan.HDBSCAN'
    dtype = np.float32

    labels_ = CumlArrayDescriptor()
    probabilities_ = CumlArrayDescriptor()
    outlier_scores_ = CumlArrayDescriptor()
    cluster_persistence_ = CumlArrayDescriptor()

    _hyperparam_interop_translator = {
        "metric": {
            "manhattan": "NotImplemented",
            "chebyshev": "NotImplemented",
            "minkowski": "NotImplemented",
        },
        "algorithm": {
            "auto": "brute",
            "ball_tree": "NotImplemented",
            "kd_tree": "NotImplemented",
        },
    }

    @device_interop_preparation
    def __init__(self, *,
                 min_cluster_size=5,
                 min_samples=None,
                 cluster_selection_epsilon=0.0,
                 max_cluster_size=0,
                 metric='euclidean',
                 alpha=1.0,
                 p=2,
                 cluster_selection_method='eom',
                 allow_single_cluster=False,
                 gen_min_span_tree=False,
                 handle=None,
                 verbose=False,
                 connectivity='knn',
                 output_type=None,
                 prediction_data=False):

        super().__init__(handle=handle,
                         verbose=verbose,
                         output_type=output_type)

        if min_samples is None:
            min_samples = min_cluster_size

        if connectivity not in ["knn", "pairwise"]:
            raise ValueError("'connectivity' can only be one of "
                             "{'knn', 'pairwise'}")

        if 2 < min_samples and min_samples > 1023:
            raise ValueError("'min_samples' must be a positive number "
                             "between 2 and 1023")

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.max_cluster_size = max_cluster_size
        self.metric = metric
        self.p = p
        self.alpha = alpha
        self.cluster_selection_method = cluster_selection_method
        self.allow_single_cluster = allow_single_cluster
        self.connectivity = connectivity
        self.gen_min_span_tree = gen_min_span_tree
        self.prediction_data = prediction_data

        self.fit_called_ = False
        self.n_clusters_ = None
        self.n_leaves_ = None
        self.core_dists = None

    @property
    def condensed_tree_(self):
        if self.condensed_tree_obj is None:

            self.condensed_tree_obj = _build_condensed_tree_plot_host(
                self.condensed_parent_.to_output("numpy"),
                self.condensed_child_.to_output("numpy"),
                self.condensed_lambdas_.to_output("numpy"),
                self.condensed_sizes_.to_output("numpy"),
                self.cluster_selection_method, self.allow_single_cluster)

        return self.condensed_tree_obj

    @condensed_tree_.setter
    def condensed_tree_(self, new_val):
        self.condensed_tree_obj = new_val

    @property
    def single_linkage_tree_(self):
        if self.single_linkage_tree_obj is None:
            hdbscan = import_hdbscan()

            with cuml.using_output_type("numpy"):
                raw_tree = np.column_stack(
                    (self.children_[0, :self.n_leaves_-1],
                     self.children_[1, :self.n_leaves_-1],
                     self.lambdas_[:self.n_leaves_-1],
                     self.sizes_[:self.n_leaves_-1]))

            raw_tree = raw_tree.astype(np.float64)

            self.single_linkage_tree_obj = hdbscan.plots.SingleLinkageTree(raw_tree)

        return self.single_linkage_tree_obj

    @single_linkage_tree_.setter
    def single_linkage_tree_(self, new_val):
        self.single_linkage_tree_obj = new_val

    @cached_property
    def prediction_data_(self):
        if not self.prediction_data:
            raise ValueError(
               'Train model with fit(prediction_data=True). or call '
               'model.generate_prediction_data()')

        hdbscan = import_hdbscan()
        from sklearn.neighbors import BallTree, KDTree

        if self.metric in KDTree.valid_metrics:
            tree_type = "kdtree"
        elif self.metric in BallTree.valid_metrics:
            tree_type = "balltree"
        else:
            raise AttributeError(f"Metric {self.metric} not supported for prediction data")

        return hdbscan.prediction.PredictionData(
            self.X_m.to_output("numpy"),
            self.condensed_tree_,
            self.min_samples or self.min_cluster_size,
            tree_type=tree_type,
            metric=self.metric
        )

    @prediction_data_.setter
    def prediction_data_(self, new_val):
        self.prediction_data_obj = new_val

    @enable_device_interop
    def generate_prediction_data(self):
        """
        Create data that caches intermediate results used for predicting
        the label of new/unseen points. This data is only useful if you
        are intending to use functions from hdbscan.prediction.
        """
        if not self.fit_called_:
            raise ValueError("The model is not trained yet (call fit() first).")

        with cuml.using_output_type("cuml"):
            labels = self.labels_

        prediction_data = _PredictionData(self.handle, self.n_rows, self.n_cols, self.core_dists)

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        cdef CondensedHierarchy[int, float] *condensed_tree
        if hasattr(self, "_condensed_hierarchy"):
            condensed_tree = (<_CondensedHierarchy>self._condensed_hierarchy).ptr
        else:
            condensed_tree = &((<_HDBSCANOutput>self._hdbscan_output).ptr.get_condensed_tree())

        lib.generate_prediction_data(
            handle_[0],
            deref(condensed_tree),
            <int*><uintptr_t>labels.ptr,
            <int*><uintptr_t>(self.inverse_label_map.ptr),
            <int> self.n_clusters_,
            deref(prediction_data.ptr),
        )
        self.handle.sync()

        self._prediction_data = prediction_data
        self.prediction_data = True

    @generate_docstring()
    @enable_device_interop
    def fit(self, X, y=None, *, convert_dtype=True) -> "HDBSCAN":
        """
        Fit HDBSCAN model from features.
        """
        X_m, n_rows, n_cols, _ = input_to_cuml_array(
            X,
            order='C',
            check_dtype=[np.float32],
            convert_to_dtype=np.float32 if convert_dtype else None,
        )
        self.X_m = X_m
        self.n_rows = n_rows
        self.n_cols = n_cols

        if cuml.accel.enabled():
            self._raw_data = self.X_m.to_output("numpy")

        cdef uintptr_t _input_ptr = X_m.ptr

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        # Validate and prepare hyperparameters
        cdef HDBSCANParams params
        params.min_samples = self.min_samples
        params.alpha = self.alpha
        params.min_cluster_size = self.min_cluster_size
        params.max_cluster_size = self.max_cluster_size
        params.cluster_selection_epsilon = self.cluster_selection_epsilon
        params.allow_single_cluster = self.allow_single_cluster

        if self.connectivity not in {"knn", "pairwise"}:
            raise ValueError(
                "`connectivity` must be one of {'knn', 'pairwise'}, "
                f"got {self.connectivity!r}"
            )

        if self.cluster_selection_method == 'eom':
            params.cluster_selection_method = CLUSTER_SELECTION_METHOD.EOM
        elif self.cluster_selection_method == 'leaf':
            params.cluster_selection_method = CLUSTER_SELECTION_METHOD.LEAF
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

        # Allocate output structures
        output = _HDBSCANOutput(self.handle, X_m)
        core_dists = CumlArray.empty(n_rows, dtype="float32")
        cdef uintptr_t core_dists_ptr = core_dists.ptr

        # Execute fit
        lib.hdbscan(
            handle_[0],
            <float*>_input_ptr,
            <int> n_rows,
            <int> n_cols,
            metric,
            params,
            deref(output.ptr),
            <float*> core_dists_ptr
        )

        # Store state on model
        self.core_dists = core_dists
        self._all_finite = True
        self.fit_called_ = True
        self.n_connected_components_ = 1
        self.n_leaves_ = n_rows

        self._hdbscan_output = output
        self.n_clusters_ = output.get_n_clusters()
        self.cluster_persistence_ = output.get_cluster_persistence()
        self.inverse_label_map = output.get_inverse_label_map()

        if self.prediction_data:
            self.generate_prediction_data()

        self.handle.sync()

        if self.gen_min_span_tree:
            self._min_spanning_tree = np.column_stack(
                (
                    output.mst_src.to_output("numpy"),
                    output.mst_dst.to_output("numpy"),
                    output.mst_weights.to_output("numpy"),
                )
            ).astype(np.float64)

        return self

    @property
    def minimum_spanning_tree_(self):
        if self._min_spanning_tree is not None:
            hdbscan = import_hdbscan()
            return hdbscan.plots.MinimumSpanningTree(
                self._min_spanning_tree,
                self._raw_data,
            )
        raise AttributeError(
            "No minimum spanning tree was generated. Set `gen_min_span_tree=True` and refit."
        )

    @generate_docstring(return_values={'name': 'preds',
                                       'type': 'dense',
                                       'description': 'Cluster indexes',
                                       'shape': '(n_samples, 1)'})
    @enable_device_interop
    def fit_predict(self, X, y=None) -> CumlArray:
        """
        Fit the HDBSCAN model from features and return
        cluster labels.
        """
        return self.fit(X).labels_

    def _extract_clusters(self, condensed_tree):
        parents, _n_edges, _, _ = \
            input_to_cuml_array(condensed_tree.to_numpy()['parent'],
                                order='C',
                                convert_to_dtype=np.int32)

        children, _, _, _ = \
            input_to_cuml_array(condensed_tree.to_numpy()['child'],
                                order='C',
                                convert_to_dtype=np.int32)

        lambdas, _, _, _ = \
            input_to_cuml_array(condensed_tree.to_numpy()['lambda_val'],
                                order='C',
                                convert_to_dtype=np.float32)

        sizes, _, _, _ = \
            input_to_cuml_array(condensed_tree.to_numpy()['child_size'],
                                order='C',
                                convert_to_dtype=np.int32)

        n_leaves = int(condensed_tree.to_numpy()['parent'].min())

        self.labels_test = CumlArray.empty(n_leaves, dtype="int32")
        self.probabilities_test = CumlArray.empty(n_leaves, dtype="float32")

        cdef uintptr_t _labels_ptr = self.labels_test.ptr
        cdef uintptr_t _parents_ptr = parents.ptr
        cdef uintptr_t _children_ptr = children.ptr
        cdef uintptr_t _sizes_ptr = sizes.ptr
        cdef uintptr_t _lambdas_ptr = lambdas.ptr
        cdef uintptr_t _probabilities_ptr = self.probabilities_test.ptr

        if self.cluster_selection_method == 'eom':
            cluster_selection_method = CLUSTER_SELECTION_METHOD.EOM
        elif self.cluster_selection_method == 'leaf':
            cluster_selection_method = CLUSTER_SELECTION_METHOD.LEAF
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        lib._extract_clusters(handle_[0],
                              <size_t> n_leaves,
                              <int> _n_edges,
                              <int*> _parents_ptr,
                              <int*> _children_ptr,
                              <float*> _lambdas_ptr,
                              <int*> _sizes_ptr,
                              <int*> _labels_ptr,
                              <float*> _probabilities_ptr,
                              <CLUSTER_SELECTION_METHOD> cluster_selection_method,
                              <bool> self.allow_single_cluster,
                              <int> self.max_cluster_size,
                              <float> self.cluster_selection_epsilon)

    def __getstate__(self):
        # TODO
        pass

    def __setstate__(self, state):
        # TODO
        pass

    def _prep_cpu_to_gpu_prediction(self, convert_dtype=True):
        """
        This is an internal function, to be called when HDBSCAN
        is trained on CPU but GPU inference is desired.
        """
        # TODO
        pass

    def gpu_to_cpu(self):
        super().gpu_to_cpu()

        # set non array hdbscan variables
        self._cpu_model.condensed_tree_ = \
            self.condensed_tree_._raw_tree
        self._cpu_model.single_linkage_tree_ = \
            self.single_linkage_tree_._linkage
        if hasattr(self, "_raw_data"):
            self._cpu_model._raw_data = self._raw_data
        if self.gen_min_span_tree:
            self._cpu_model.minimum_spanning_tree_ = \
                self.minimum_spanning_tree_._mst

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + [
            "metric",
            "min_cluster_size",
            "max_cluster_size",
            "min_samples",
            "cluster_selection_epsilon",
            "cluster_selection_method",
            "p",
            "allow_single_cluster",
            "connectivity",
            "alpha",
            "gen_min_span_tree",
            "prediction_data"
        ]

    def get_attr_names(self):
        attr_names = ['labels_', 'probabilities_', 'cluster_persistence_',
                      'condensed_tree_', 'single_linkage_tree_',
                      'outlier_scores_', '_all_finite']
        if self.gen_min_span_tree:
            attr_names = attr_names + ['minimum_spanning_tree_']
        if self.prediction_data:
            attr_names = attr_names + ['prediction_data_']

        return attr_names

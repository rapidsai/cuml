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
    return state.get_condensed_tree()


def _cupy_array_from_ptr(ptr, shape, dtype, owner):
    dtype = np.dtype(dtype)
    mem = cp.cuda.UnownedMemory(
        ptr=ptr,
        size=np.prod(shape) * dtype.itemsize,
        owner=owner,
        device_id=-1,
    )
    mem_ptr = cp.cuda.memory.MemoryPointer(mem, 0)
    return cp.ndarray(shape=shape, dtype=dtype, memptr=mem_ptr)


cdef class _HDBSCANState:
    cdef lib.hdbscan_output *hdbscan_output
    cdef lib.CondensedHierarchy[int, float] *condensed_tree
    cdef lib.PredictionData[int, float] *prediction_data
    cdef int n_clusters
    cdef object core_dists
    cdef object inverse_label_map

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

    cdef lib.CondensedHierarchy[int, float]* get_condensed_tree_ptr(self):
        if self.hdbscan_output != NULL:
            return &(self.hdbscan_output.get_condensed_tree())
        return self.condensed_tree

    @staticmethod
    def from_dendrogram(dendrogram, min_cluster_size):
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
            check_dtype=[np.int32],
            convert_to_dtype=np.int32,
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
            check_dtype=[np.int32],
            convert_to_dtype=np.int32,
        )[0]

        cdef size_t n_leaves = dendrogram.shape[0] + 1

        handle = Handle()
        cdef handle_t *handle_ = <handle_t*> <size_t> handle.getHandle()

        self.condensed_tree = new lib.CondensedHierarchy[int, float](handle_[0], n_leaves)
        lib.build_condensed_hierarchy(
            handle_[0],
            <int*><uintptr_t>(children.ptr),
            <float*><uintptr_t>(lambdas.ptr),
            <int*><uintptr_t>(sizes.ptr),
            <int>min_cluster_size,
            n_leaves,
            deref(self.condensed_tree)
        )
        return self

    @staticmethod
    cdef init_and_fit(
        handle,
        X,
        HDBSCANParams params,
        DistanceType metric,
        bool gen_min_span_tree,
    ):
        cdef _HDBSCANState self = _HDBSCANState.__new__(_HDBSCANState)

        cdef int n_rows = X.shape[0]
        cdef int n_cols = X.shape[1]

        # Allocate output structures
        labels = CumlArray.empty(n_rows, dtype="int32", index=X.index)
        probabilities = CumlArray.empty(n_rows, dtype="float32")

        children = CumlArray.empty((2, n_rows), dtype="int32")
        lambdas = CumlArray.empty(n_rows, dtype="float32")
        sizes = CumlArray.empty(n_rows, dtype="int32")

        mst_src = CumlArray.empty(n_rows - 1, dtype="int32")
        mst_dst = CumlArray.empty(n_rows - 1, dtype="int32")
        mst_weights = CumlArray.empty(n_rows - 1, dtype="float32")

        core_dists = CumlArray.empty(n_rows, dtype="float32")

        cdef handle_t* handle_ = <handle_t*><uintptr_t>handle.getHandle()

        self.hdbscan_output = new lib.hdbscan_output(
            handle_[0],
            n_rows,
            <int*><uintptr_t>(labels.ptr),
            <float*><uintptr_t>(probabilities.ptr),
            <int*><uintptr_t>(children.ptr),
            <int*><uintptr_t>(sizes.ptr),
            <float*><uintptr_t>(lambdas.ptr),
            <int*><uintptr_t>(mst_src.ptr),
            <int*><uintptr_t>(mst_dst.ptr),
            <float*><uintptr_t>(mst_weights.ptr)
        )

        # Execute fit
        lib.hdbscan(
            handle_[0],
            <float*><uintptr_t>(X.ptr),
            n_rows,
            n_cols,
            metric,
            params,
            deref(self.hdbscan_output),
            <float*><uintptr_t>(core_dists.ptr),
        )
        handle.sync()

        # Extract and store local state
        self.n_clusters = self.hdbscan_output.get_n_clusters()
        if self.n_clusters > 0:
            self.inverse_label_map = _cupy_array_from_ptr(
                <size_t>self.hdbscan_output.get_inverse_label_map(),
                (self.n_clusters,),
                np.int32,
                self
            ).copy()
        else:
            self.inverse_label_map = cp.empty((0,), dtype=np.int32)
        self.core_dists = core_dists

        # Extract and prepare results
        if self.n_clusters > 0:
            cluster_persistence = CumlArray(
                data=_cupy_array_from_ptr(
                    <size_t>self.hdbscan_output.get_stabilities(),
                    (1, self.n_clusters),
                    np.float32,
                    self,
                )
            )
        else:
            cluster_persistence = CumlArray.empty((0,), dtype="float32")

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
        if self.prediction_data != NULL:
            return

        cdef int n_rows = X.shape[0]
        cdef int n_cols = X.shape[0]
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

        self.prediction_data = new lib.PredictionData[int, float](
            handle_[0],
            n_rows,
            n_cols,
            <float*><uintptr_t>(self.core_dists.ptr),
        )

        cdef lib.CondensedHierarchy[int, float] *condensed_tree = self.get_condensed_tree_ptr()

        lib.generate_prediction_data(
            handle_[0],
            deref(condensed_tree),
            <int*><uintptr_t>(labels.ptr),
            <int*><uintptr_t>(self.inverse_label_map.data.ptr),
            <int> self.n_clusters,
            deref(self.prediction_data),
        )
        handle.sync()

    def get_condensed_tree(self):
        cdef lib.CondensedHierarchy[int, float]* condensed_tree = self.get_condensed_tree_ptr()

        n_condensed_tree_edges = condensed_tree.get_n_edges()

        tree = np.recarray(
            shape=(n_condensed_tree_edges,),
            formats=[np.int32, np.int32, np.float32, np.int32],
            names=('parent', 'child', 'lambda_val', 'child_size'),
        )

        parents = _cupy_array_from_ptr(
            <size_t>condensed_tree.get_parents(),
            (n_condensed_tree_edges,),
            np.int32,
            self,
        )

        children = _cupy_array_from_ptr(
            <size_t>condensed_tree.get_children(),
            (n_condensed_tree_edges,),
            np.int32,
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
            np.int32,
            self,
        )

        tree['parent'] = parents.get()
        tree['child'] = children.get()
        tree['lambda_val'] = lambdas.get()
        tree['child_size'] = sizes.get()

        return tree


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

    labels_ = CumlArrayDescriptor()
    probabilities_ = CumlArrayDescriptor()
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

    @property
    def dtype(self):
        return np.float32

    @property
    def _raw_data(self):
        return self.X_m.to_output("numpy")

    @cached_property
    def _condensed_tree(self):
        return self._state.get_condensed_tree()

    @property
    def condensed_tree_(self):
        hdbscan = import_hdbscan()
        return hdbscan.plots.CondensedTree(
            self._condensed_tree,
            self.cluster_selection_method,
            self.allow_single_cluster
        )

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

    @property
    def single_linkage_tree_(self):
        hdbscan = import_hdbscan()
        return hdbscan.plots.SingleLinkageTree(self._single_linkage_tree)

    @cached_property
    def prediction_data_(self):
        if not self.prediction_data:
            raise ValueError(
               "Train model with fit(prediction_data=True). or call "
               "model.generate_prediction_data()"
            )

        hdbscan = import_hdbscan()
        from sklearn.neighbors import BallTree, KDTree

        if self.metric in KDTree.valid_metrics:
            tree_type = "kdtree"
        elif self.metric in BallTree.valid_metrics:
            tree_type = "balltree"
        else:
            raise AttributeError(f"Metric {self.metric} not supported for prediction data")

        return hdbscan.prediction.PredictionData(
            self._raw_data,
            self.condensed_tree_,
            self.min_samples or self.min_cluster_size,
            tree_type=tree_type,
            metric=self.metric
        )

    @enable_device_interop
    def generate_prediction_data(self):
        """
        Create data that caches intermediate results used for predicting
        the label of new/unseen points. This data is only useful if you
        are intending to use functions from hdbscan.prediction.
        """
        if not hasattr(self, "labels_"):
            raise ValueError("The model is not trained yet (call fit() first).")

        with cuml.using_output_type("cuml"):
            labels = self.labels_

        self._state.generate_prediction_data(self.handle, self.X_m, labels)
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
            X_m,
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
        self._all_finite = True
        self.n_connected_components_ = 1
        self.n_leaves_ = n_rows

        if self.prediction_data:
            self.generate_prediction_data()

        return self

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

    def __getstate__(self):
        # TODO
        pass

    def __setstate__(self, state):
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

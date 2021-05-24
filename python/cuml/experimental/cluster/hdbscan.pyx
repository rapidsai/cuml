#
# Copyright (c) 2021, NVIDIA CORPORATION.
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

from libc.stdint cimport uintptr_t
from libcpp cimport bool

from cython.operator cimport dereference as deref

import numpy as np
import cupy as cp

from cuml.common.array import CumlArray
from cuml.common.base import Base
from cuml.common.doc_utils import generate_docstring
from cuml.raft.common.handle cimport handle_t
from cuml.common import input_to_cuml_array
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.mixins import ClusterMixin
from cuml.common.mixins import CMajorInputTagMixin
from cuml.common import logger

import cuml
from cuml.metrics.distance_type cimport DistanceType


cdef extern from "cuml/cluster/hdbscan.hpp" namespace "ML::HDBSCAN::Common":

    ctypedef enum CLUSTER_SELECTION_METHOD:
        EOM "ML::HDBSCAN::Common::CLUSTER_SELECTION_METHOD::EOM"
        LEAF "ML::HDBSCAN::Common::CLUSTER_SELECTION_METHOD::LEAF"

    cdef cppclass CondensedHierarchy_int_float:
        int *get_parents()
        int *get_children()
        float *get_lambdas()
        int *get_sizes()
        int get_n_edges()

    cdef cppclass hdbscan_output[int, float]:
        hdbscan_output(const handle_t &handle,
                       int n_leaves,
                       int *labels,
                       float *probabilities,
                       int *children,
                       int *sizes,
                       float *deltas,
                       int *mst_src,
                       int *mst_dst,
                       float *mst_weights)
        int get_n_leaves()
        int get_n_clusters()
        float *get_stabilities()
        CondensedHierarchy_int_float &get_condensed_tree()

    cdef cppclass HDBSCANParams:
        int k
        int min_samples
        int min_cluster_size
        int max_cluster_size,

        float cluster_selection_epsilon,

        bool allow_single_cluster,
        CLUSTER_SELECTION_METHOD cluster_selection_method


cdef extern from "cuml/cluster/hdbscan.hpp" namespace "ML":

    void hdbscan(const handle_t & handle,
                 const float * X,
                 size_t m, size_t n,
                 DistanceType metric,
                 HDBSCANParams & params,
                 hdbscan_output & output)

_metrics_mapping = {
    'l1': DistanceType.L1,
    'cityblock': DistanceType.L1,
    'manhattan': DistanceType.L1,
    'l2': DistanceType.L2SqrtExpanded,
    'euclidean': DistanceType.L2SqrtExpanded,
    'cosine': DistanceType.CosineExpanded
}


def delete_hdbscan_output(obj):
    cdef hdbscan_output *output
    if hasattr(obj, "hdbscan_output_"):
        output = <hdbscan_output*>\
                  <uintptr_t> obj.hdbscan_output_
        del output
        del obj.hdbscan_output_


class HDBSCAN(Base, ClusterMixin, CMajorInputTagMixin):

    """
    HDBSCAN Clustering

    Recursively merges the pair of clusters that minimally increases a
    given linkage distance.

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
        See [2]_ for more information.

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
        See [3]_ for more information. Note that this should not be used
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

    metric : string or callable, optional (default='minkowski')
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string or callable, it must be one of
        the options allowed by metrics.pairwise.pairwise_distances for its
        metric parameter.
        If metric is "precomputed", X is assumed to be a distance matrix and
        must be square.

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
        Whether to populate the minimum_spanning_tree_ member for
        utilizing plotting tools. This requires the `hdbscan` CPU Python
        package to be installed.

    gen_condensed_tree : bool, optional (default=False)
        Whether to populate the condensed_tree_ member for
        utilizing plotting tools. This requires the `hdbscan` CPU
        Python package to be installed.

    gen_single_linkage_tree_ : bool, optinal (default=False)
        Whether

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
        scores can be guage the relative coherence of the clusters output
        by the algorithm.

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
    outlier_scores_ = CumlArrayDescriptor()
    probabilities_ = CumlArrayDescriptor()

    # Single Linkage Tree
    children_ = CumlArrayDescriptor()
    lambdas_ = CumlArrayDescriptor()
    sizes_ = CumlArrayDescriptor()

    # Minimum Spanning Tree
    mst_src_ = CumlArrayDescriptor()
    mst_dst_ = CumlArrayDescriptor()
    mst_weights_ = CumlArrayDescriptor()

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
                 n_neighbors=10,
                 output_type=None):

        super().__init__(handle,
                         verbose,
                         output_type)

        if min_samples is None:
            min_samples = min_cluster_size

        if connectivity not in ["knn", "pairwise"]:
            raise ValueError("'connectivity' can only be one of "
                             "{'knn', 'pairwise'}")

        if n_neighbors > 1023 or n_neighbors < 2:
            raise ValueError("'n_neighbors' must be a positive number "
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
        self.n_neighbors = n_neighbors
        self.connectivity = connectivity

        self.fit_called_ = False

        self.n_clusters_ = None
        self.n_leaves_ = None

        self._condensed_tree = None
        self._single_linkage_tree = None

        self.gen_min_span_tree_ = gen_min_span_tree

    def _build_condensed_tree(self):

        if not self.fit_called_:
            return None

        if self._condensed_tree is None:
            raw_tree = np.recarray(shape=(self.condensed_parent_.shape[0],),
                                   formats=[np.intp, np.intp, float, np.intp],
                                   names=('parent', 'child', 'lambda_val',
                                          'child_size'))
            raw_tree['parent'] = self.condensed_parent_
            raw_tree['child'] = self.condensed_child_
            raw_tree['lambda_val'] = self.condensed_lambdas_
            raw_tree['child_size'] = self.condensed_sizes_

        try:
            from hdbscan.plots import CondensedTree
        except Exception as e:
            raise ImportError("hdbscan must be installed to use plots")

        return CondensedTree(raw_tree,
                             self.cluster_selection_epsilon,
                             self.allow_single_cluster)

    def _build_single_linkage_tree(self):

        if not self.fit_called_:
            return None

        if self._single_linkage_tree is None:

            with cuml.using_output_type("numpy"):
                raw_tree = np.column_stack(
                    (self.children_[0, :self.n_leaves_-1],
                     self.children_[1, :self.n_leaves_-1],
                     self.lambdas_[:self.n_leaves_-1],
                     self.sizes_[:self.n_leaves_-1]))

            raw_tree = raw_tree.astype(np.float64)

        try:
            from hdbscan.plots import SingleLinkageTree
        except Exception as e:
            raise ImportError("hdbscan must be installed "
                              "to use plots")

        return SingleLinkageTree(raw_tree)

    def _build_minimum_spanning_tree(self, X):

        with cuml.using_output_type("numpy"):
            raw_tree = np.column_stack((self.mst_src_,
                                        self.mst_dst_,
                                        self.mst_weights_))

        raw_tree = raw_tree.astype(np.float64)

        try:
            from hdbscan.plots import MinimumSpanningTree
        except Exception as e:
            raise ImportError("hdbscan must be installed to use plots")

        self.minimum_spanning_tree_ = MinimumSpanningTree(
            raw_tree, X.to_output("numpy"))

    def __dealloc__(self):
        delete_hdbscan_output(self)

    def _cuml_array_from_ptr(self, ptr, buf_size, shape, dtype):

        mem = cp.cuda.UnownedMemory(ptr=ptr, size=buf_size,
                                    owner=self.hdbscan_output_,
                                    device_id=-1)
        mem_ptr = cp.cuda.memory.MemoryPointer(mem, 0)

        return CumlArray(data=cp.ndarray(shape=shape,
                                         dtype=dtype,
                                         memptr=mem_ptr)).to_output('numpy')

    def _construct_condensed_tree_attribute(self, ptr, dtype="int32"):
        cdef hdbscan_output *hdbscan_output_ = \
                <hdbscan_output*><size_t>self.hdbscan_output_

        n_condensed_tree_edges = \
            hdbscan_output_.get_condensed_tree().get_n_edges()

        return self._cuml_array_from_ptr(
            ptr, n_condensed_tree_edges * sizeof(float),
            (n_condensed_tree_edges,), dtype
        )

    def _construct_output_attributes(self):

        cdef hdbscan_output *hdbscan_output_ = \
                <hdbscan_output*><size_t>self.hdbscan_output_

        self.n_clusters_ = hdbscan_output_.get_n_clusters()

        self.cluster_persistence_ = self._cuml_array_from_ptr(
            <size_t>hdbscan_output_.get_stabilities(),
            hdbscan_output_.get_n_clusters() * sizeof(float),
            (1, hdbscan_output_.get_n_clusters()), "float32"
        )

        self.condensed_parent_ = self._construct_condensed_tree_attribute(
            <size_t>hdbscan_output_.get_condensed_tree().get_parents())

        self.condensed_child_ = self._construct_condensed_tree_attribute(
            <size_t>hdbscan_output_.get_condensed_tree().get_children())

        self.condensed_lambdas_ = \
            self._construct_condensed_tree_attribute(
                <size_t>hdbscan_output_.get_condensed_tree().get_lambdas(),
                "float32")

        self.condensed_sizes_ = self._construct_condensed_tree_attribute(
            <size_t>hdbscan_output_.get_condensed_tree().get_sizes())

    @generate_docstring()
    def fit(self, X, y=None, convert_dtype=True) -> "HDBSCAN":
        """
        Fit HDBSCAN model from features.
        """

        X_m, n_rows, n_cols, self.dtype = \
            input_to_cuml_array(X, order='C',
                                check_dtype=[np.float32],
                                convert_to_dtype=(np.float32
                                                  if convert_dtype
                                                  else None))

        cdef uintptr_t input_ptr = X_m.ptr

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        # Hardcode n_components_ to 1 for single linkage. This will
        # not be the case for other linkage types.
        self.n_connected_components_ = 1
        self.n_leaves_ = n_rows

        self.labels_ = CumlArray.empty(n_rows, dtype="int32")
        self.children_ = CumlArray.empty((2, n_rows), dtype="int32")
        self.probabilities_ = CumlArray.empty(n_rows, dtype="float32")
        self.sizes_ = CumlArray.empty(n_rows, dtype="int32")
        self.lambdas_ = CumlArray.empty(n_rows, dtype="float32")
        self.mst_src_ = CumlArray.empty(n_rows-1, dtype="int32")
        self.mst_dst_ = CumlArray.empty(n_rows-1, dtype="int32")
        self.mst_weights_ = CumlArray.empty(n_rows-1, dtype="float32")

        cdef uintptr_t labels_ptr = self.labels_.ptr
        cdef uintptr_t children_ptr = self.children_.ptr
        cdef uintptr_t sizes_ptr = self.sizes_.ptr
        cdef uintptr_t lambdas_ptr = self.lambdas_.ptr
        cdef uintptr_t probabilities_ptr = self.probabilities_.ptr
        cdef uintptr_t mst_src_ptr = self.mst_src_.ptr
        cdef uintptr_t mst_dst_ptr = self.mst_dst_.ptr
        cdef uintptr_t mst_weights_ptr = self.mst_weights_.ptr

        # If calling fit a second time, release
        # any memory owned from previous trainings
        delete_hdbscan_output(self)

        cdef hdbscan_output *linkage_output = new hdbscan_output(
            handle_[0], n_rows,
            <int*>labels_ptr,
            <float*>probabilities_ptr,
            <int*>children_ptr,
            <int*>sizes_ptr,
            <float*>lambdas_ptr,
            <int*>mst_src_ptr,
            <int*>mst_dst_ptr,
            <float*>mst_weights_ptr)

        self.hdbscan_output_ = <size_t>linkage_output

        cdef HDBSCANParams params
        params.k = self.n_neighbors
        params.min_samples = self.min_samples
        # params.alpha = self.alpha
        params.min_cluster_size = self.min_cluster_size
        params.max_cluster_size = self.max_cluster_size
        params.cluster_selection_epsilon = self.cluster_selection_epsilon
        params.allow_single_cluster = self.allow_single_cluster

        if self.cluster_selection_method == 'eom':
            params.cluster_selection_method = CLUSTER_SELECTION_METHOD.EOM
        elif self.cluster_selection_method == 'leaf':
            params.cluster_selection_method = CLUSTER_SELECTION_METHOD.LEAF
        else:
            raise ValueError("Cluster selection method not supported. "
                             "Must one of {'eom', 'leaf'}")

        cdef DistanceType metric
        if self.metric in _metrics_mapping:
            metric = _metrics_mapping[self.metric]
        else:
            raise ValueError("'affinity' %s not supported." % self.affinity)

        if self.connectivity == 'knn':
            hdbscan(handle_[0],
                    <float*>input_ptr,
                    <int> n_rows,
                    <int> n_cols,
                    <DistanceType> metric,
                    params,
                    deref(linkage_output))
        else:
            raise ValueError("'connectivity' can only be one of "
                             "{'knn', 'pairwise'}")

        self.handle.sync()

        self.fit_called_ = True

        self._construct_output_attributes()

        try:

            from hdbscan.plots import SingleLinkageTree

            if self.gen_min_span_tree_:
                self._minimum_spanning_tree = \
                    self._build_minimum_spanning_tree(X_m)

            self._condensed_tree = self._build_condensed_tree()
            self._single_linkage_tree = self._build_single_linkage_tree()

        except Exception as e:
            logger.warn("hdbscan must be installed to use plots")

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

    def get_param_names(self):
        return super().get_param_names() + [
            "n_neighbors",
            "metric",
            "min_cluster_size",
            "max_cluster_size",
            "min_samples",
            "cluster_selection_epsilon",
            "cluster_selection_method",
            "p",
            "allow_single_cluster",
            "connectivity",
            "n_neighbors"
        ]

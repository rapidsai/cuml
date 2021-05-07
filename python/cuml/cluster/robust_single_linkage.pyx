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

from cuml.metrics.distance_type cimport DistanceType

from .hdbscan_plot import SingleLinkageTree

cdef extern from "cuml/cluster/hdbscan.hpp" namespace "ML::HDBSCAN::Common":
    cdef cppclass robust_single_linkage_output[int, float]:
        robust_single_linkage_output(const handle_t &handle,
                       int n_leaves,
                       int *labels,
                       int *children,
                       int *sizes,
                       float *deltas,
                       int *mst_src,
                       int *mst_dst,
                       float *mst_weights)
        int get_n_leaves()
        int get_n_clusters()

    cdef cppclass RobustSingleLinkageParams:
        int k
        int min_samples
        int min_cluster_size

        float alpha

        float cluster_selection_epsilon,

        bool allow_single_cluster,


cdef extern from "cuml/cluster/hdbscan.hpp" namespace "ML":

    void robust_single_linkage(const handle_t &handle,
                               const float *X,
                               size_t m,
                               size_t n,
                               DistanceType metric,
                               RobustSingleLinkageParams &params,
                               robust_single_linkage_output &out)

def delete_output(obj):
    cdef robust_single_linkage_output *output
    if hasattr(obj, "hdbscan_output_"):
        output = <robust_single_linkage_output*>\
                  <uintptr_t> obj.rsl_output_
        del output
        del obj.rsl_output_


_metrics_mapping = {
    'l1': DistanceType.L1,
    'cityblock': DistanceType.L1,
    'manhattan': DistanceType.L1,
    'l2': DistanceType.L2SqrtExpanded,
    'euclidean': DistanceType.L2SqrtExpanded,
    'cosine': DistanceType.CosineExpanded
}


class RobustSingleLinkage(Base, ClusterMixin, CMajorInputTagMixin):
    """Perform robust single linkage clustering from a vector array
    or distance matrix.
    Robust single linkage is a modified version of single linkage that
    attempts to be more robust to noise. Specifically the goal is to
    more accurately approximate the level set tree of the unknown
    probability density function from which the sample data has
    been drawn.

    Parameters
    ----------
    X : array of shape (n_samples, n_features), or \
            array of shape (n_samples, n_samples)
        A feature array
    cut : float
        The reachability distance value to cut the cluster heirarchy at
        to derive a flat cluster labelling.
    k : int, optional (default=5)
        Reachability distances will be computed with regard to the `k`
        nearest neighbors.
    alpha : float, optional (default=np.sqrt(2))
        Distance scaling for reachability distance computation. Reachability
        distance is computed as
        $max \{ core_k(a), core_k(b), 1/\alpha d(a,b) \}$.
    gamma : int, optional (default=5)
        Ignore any clusters in the flat clustering with size less than gamma,
        and declare points in such clusters as noise points.
    metric : string, optional (default='euclidean')
        The metric to use when calculating distance between instances in a
        feature array.
    metric_params : dict, option (default={})
        Keyword parameter arguments for calling the metric (for example
        the p values if using the minkowski metric).

    Attributes
    -------
    labels_ : ndarray, shape (n_samples, )
        Cluster labels for each point.  Noisy samples are given the label -1.
    cluster_hierarchy_ : SingleLinkageTree object
        The single linkage tree produced during clustering.
        This object provides several methods for:
            * Plotting
            * Generating a flat clustering
            * Exporting to NetworkX
            * Exporting to Pandas

    References
    ----------
    .. [1] Chaudhuri, K., & Dasgupta, S. (2010). Rates of convergence for the
       cluster tree. In Advances in Neural Information Processing Systems
       (pp. 343-351).

    """

    labels_ = CumlArrayDescriptor()

    # Single Linkage Tree
    children_ = CumlArrayDescriptor()
    lambdas_ = CumlArrayDescriptor()
    sizes_ = CumlArrayDescriptor()

    # Minimum Spanning Tree
    mst_src_ = CumlArrayDescriptor()
    mst_dst_ = CumlArrayDescriptor()
    mst_weights_ = CumlArrayDescriptor()

    def __init__(self, cut=0.4, k=5, alpha=1.4142135623730951, gamma=5,
                 metric='euclidean', metric_params={}):

        self.cut = cut
        self.k = k
        self.alpha = alpha
        self.gamma = gamma
        self.metric = metric
        self.metric_params = metric_params

    def _construct_output_attributes(self):

        cdef robust_single_linkage_output *rsl_output = \
                <robust_single_linkage_output*><size_t>self.rsl_output_

        self.n_clusters_ = rsl_output.get_n_clusters()

    def fit(self, X, y=None, convert_dtype=-True)->"HDBSCAN":

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

        self.labels_ = CumlArray.empty(n_rows, dtype="int32")
        self.children_ = CumlArray.empty((2, n_rows), dtype="int32")
        self.sizes_ = CumlArray.empty(n_rows, dtype="int32")
        self.lambdas_ = CumlArray.empty(n_rows, dtype="float32")
        self.mst_src_ = CumlArray.empty(n_rows-1, dtype="int32")
        self.mst_dst_ = CumlArray.empty(n_rows-1, dtype="int32")
        self.mst_weights_ = CumlArray.empty(n_rows-1, dtype="float32")

        cdef uintptr_t labels_ptr = self.labels_.ptr
        cdef uintptr_t children_ptr = self.children_.ptr
        cdef uintptr_t sizes_ptr = self.sizes_.ptr
        cdef uintptr_t lambdas_ptr = self.lambdas_.ptr
        cdef uintptr_t mst_src_ptr = self.mst_src_.ptr
        cdef uintptr_t mst_dst_ptr = self.mst_dst_.ptr
        cdef uintptr_t mst_weights_ptr = self.mst_weights_.ptr

        # If calling fit a second time, release
        # any memory owned from previous trainings
        delete_output(self)

        cdef robust_single_linkage_output *linkage_output = \
            new robust_single_linkage_output(
                handle_[0], n_rows,
                <int*>labels_ptr,
                <int*>children_ptr,
                <int*>sizes_ptr,
                <float*>lambdas_ptr,
                <int*>mst_src_ptr,
                <int*>mst_dst_ptr,
                <float*>mst_weights_ptr)

        self.rsl_output_ = <size_t>linkage_output

        cdef RobustSingleLinkageParams params
        params.k = self.n_neighbors
        params.min_samples = self.min_samples
        params.alpha = self.alpha
        params.min_cluster_size = self.min_cluster_size
        params.cluster_selection_epsilon = self.cut
        params.allow_single_cluster = self.allow_single_cluster

        cdef DistanceType metric
        if self.metric in _metrics_mapping:
            metric = _metrics_mapping[self.metric]
        else:
            raise ValueError("'affinity' %s not supported." % self.affinity)

        robust_single_linkage(handle_[0],
                < float * > input_ptr,
                < int > n_rows,
                < int > n_cols,
                < DistanceType > metric,
                  params,
                  deref(linkage_output))

        self.handle.sync()

        self._construct_output_attributes()

        print("Labels: %s" % self.labels_.to_output("numpy"))

        return self

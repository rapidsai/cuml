#
# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

import ctypes
import cudf
import numpy as np
import cupy as cp

from libcpp cimport bool
from libc.stdint cimport uintptr_t, int64_t
from libc.stdlib cimport calloc, malloc, free

from cuml.common.array import CumlArray
from cuml.common.base import Base
from cuml.common.doc_utils import generate_docstring
from cuml.raft.common.handle cimport handle_t
from cuml.common import input_to_cuml_array
from cuml.common import using_output_type
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.mixins import ClusterMixin
from cuml.common.mixins import CMajorInputTagMixin

from cuml.metrics.distance_type cimport DistanceType


cdef extern from "raft/sparse/hierarchy/common.h" namespace "raft::hierarchy":

    cdef cppclass linkage_output_int_float:
        int m
        int n_clusters
        int n_leaves
        int n_connected_components
        int *labels
        int *children

cdef extern from "cuml/cluster/linkage.hpp" namespace "ML":

    cdef void single_linkage_pairwise(
        const handle_t &handle,
        const float *X,
        size_t m,
        size_t n,
        linkage_output_int_float *out,
        DistanceType metric,
        int n_clusters
    ) except +

    cdef void single_linkage_neighbors(
        const handle_t &handle,
        const float *X,
        size_t m,
        size_t n,
        linkage_output_int_float *out,
        DistanceType metric,
        int c,
        int n_clusters
    ) except +


class AgglomerativeClustering(Base, ClusterMixin, CMajorInputTagMixin):

    labels_ = CumlArrayDescriptor()
    children_ = CumlArrayDescriptor()

    def __init__(self, n_clusters=2, affinity="euclidean", linkage="single",
                 compute_distances=False, handle=None, verbose=False,
                 connectivity='knn', n_neighbors=10, output_type=None):

        """
        Agglomerative Clustering

        Recursively merges the pair of clusters that minimally increases a
        given linkage distance.

        :param n_clusters: number of clusters
        :param affinity: distance measure to use for linkage construction
        :param linkage: linkage criterion to use. Currently only 'single' is supported.
        :param compute_distances:
        :param handle:
        :param verbose:
        :param connectivity: 'knn' constructs a knn graph to save
        :param n_neighbors:
        :param output_type:
        """
        super(AgglomerativeClustering, self).__init__(handle,
                                                      verbose,
                                                      output_type)

        self.n_clusters = n_clusters
        self.affinity = affinity
        self.linkage = linkage
        self.n_neighbors = n_neighbors
        self.connectivity = connectivity

        self.labels_ = None
        self.n_clusters_ = None
        self.n_leaves_ = None
        self.n_connected_components_ = None
        self.children_ = None
        self.distances_ = None


    def fit(self, X, y=None):
        """
        Fit the hierarchical clustering from features
        :param X:
        :param y:
        :return:
        """

        X_m, n_rows, n_cols, self.dtype = \
            input_to_cuml_array(X, order='C',
                                check_dtype=[np.float32, np.float64])

        cdef uintptr_t input_ptr = X_m.ptr

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        self.labels_ = CumlArray.empty(n_rows, dtype="int32")
        self.children_ = CumlArray.empty((2, (n_rows - 1)), dtype="int32")
        cdef uintptr_t labels_ptr = self.labels_.ptr
        cdef uintptr_t children_ptr = self.children_.ptr

        cdef linkage_output_int_float* linkage_output = new linkage_output_int_float()

        linkage_output.children = <int*>children_ptr
        linkage_output.labels = <int*>labels_ptr

        cdef DistanceType metric = DistanceType.L2SqrtExpanded

        if self.connectivity == 'knn':
            single_linkage_neighbors(handle_[0],
                                     <float*>input_ptr,
                                     <int> n_rows,
                                     <int> n_cols,
                                     <linkage_output_int_float*> linkage_output,
                                     <DistanceType> metric,
                                     <int>self.n_neighbors,
                                     <int> self.n_clusters)
        elif self.connectivity == 'pairwise':
            single_linkage_pairwise(handle_[0],
                                    <float*>input_ptr,
                                    <int> n_rows,
                                    <int> n_cols,
                                    <linkage_output_int_float*> linkage_output,
                                    <DistanceType> metric,
                                    <int> self.n_clusters)
        else:
            raise ValueError("'connectivity' can be one of {'knn', 'pairwise'}")

        self.handle.sync()

    def fit_predict(self, X, y=None):
        """
        Fit the hierarchical clustering from features,
        and return cluster labels.
        :param X:
        :param y:
        :return:
        """
        return self.fit(X).labels_

    def get_param_names(self):
        return super().get_param_names() + [
            "n_clusters",
            "affinity",
            "linkage",
            "compute_distances",
            "n_neighbors"
        ]

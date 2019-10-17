#
# Copyright (c) 2019, NVIDIA CORPORATION.
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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cuml.neighbors import NearestNeighbors

import numpy as np
import pandas as pd
import cudf
import ctypes
import cuml
import warnings

from cuml.common.base import Base
from cuml.utils import get_cudf_column_ptr, get_dev_array_ptr, \
    input_to_dev_array, zeros, row_matrix

from cython.operator cimport dereference as deref

from cuml.common.handle cimport cumlHandle


from libcpp cimport bool
from libcpp.memory cimport shared_ptr

import rmm
from libc.stdlib cimport malloc, free

from libc.stdint cimport uintptr_t, int64_t
from libc.stdlib cimport calloc, malloc, free

from numba import cuda
import rmm


cimport cuml.common.handle
cimport cuml.common.cuda

cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        cppclass iterator:
            T operator*()
            iterator operator++()
            bint operator==(iterator)
            bint operator!=(iterator)
        vector()
        void push_back(T&)
        T& operator[](int)
        T& at(int)
        iterator begin()
        iterator end()
        void resize(size_t n)

cdef extern from "cumlprims/opg/matrix/data.hpp" \
    namespace "MLCommon::Matrix":

    cdef cppclass Data[T]:
        Data(T *ptr, size_t totalSize)

    cdef cppclass floatData_t:
        float *ptr
        size_t totalSize

    cdef cppclass doubleData_t:
        double *ptr
        size_t totalSize

ctypedef Data[int64_t] int64Data_t


cdef extern from "cumlprims/opg/matrix/part_descriptor.hpp" \
    namespace "MLCommon::Matrix":

    cdef cppclass RankSizePair:
        int rank
        size_t size

    cdef cppclass PartDescriptor:
        PartDescriptor(size_t M,
                       size_t N,
                       vector[RankSizePair*] &partsToRanks,
                       int myrank)



cdef extern from "cumlprims/opg/selection/knn.hpp" \
    namespace "MLCommon::Selection:opg":

    cdef brute_force_knn(
        cumlHandle &handle,
        vector[int64Data_t*] &out_I,
        vector[floatData_t*] &out_D,
        vector[floatData_t*] &idx_data,
        PartDescriptor &idx_desc,
        vector[floatData_t*] &query_data,
        PartDescriptor &query_desc,
        int k,
        size_t batch_size,
        bool verbose
    )






class NearestNeighborsMG(NearestNeighbors):
    """
    Multi-node multi-GPU Nearest Neighbors kneighbors query.

    NOTE: This implementation of NearestNeighbors is meant to be
    used with an initialized cumlCommunicator instance inside an
    existing distributed system. Refer to the Dask NearestNeighbors
     implementation in `cuml.dask.neighbors.nearest_neighbors`.
    """

    def _build_dataFloat(self, arr_interfaces):
        """
        Instantiate a container object for a float data pointer
        and size.
        :param arr_interfaces: 
        :return: 
        """
        cdef vector[floatData_t*] *dataF = new vector[floatData_t*]()
        # dataF.resize(len(arr_interfaces))

        cdef uintptr_t input_ptr
        for x_i in range(len(arr_interfaces)):
            x = arr_interfaces[x_i]
            input_ptr = x["data"]
            data = < floatData_t * > malloc(sizeof(floatData_t))
            data.ptr = < float * > input_ptr
            data.totalSize = < size_t > (x["shape"][0] * x["shape"][1] * sizeof(float))

            dataF.push_back(data)

        return <size_t>dataF

    def _freeFloatD(self, data, arr_interfaces):
        cdef uintptr_t data_ptr = data
        cdef vector[floatData_t*] *d = <vector[floatData_t*]*>data_ptr
        for x_i in range(len(arr_interfaces)):
            free(d.at(x_i))
        free(d)

    def kneighbors(self, indices, index_m, n, index_partsToRanks,
                         queries, query_m, query_partsToRanks,
                         rank):
        """
        Query the kneighbors of an index
        :param indices: [__cuda_array_interface__] of local index partitions
        :param index_m: number of total index rows
        :param n: number of columns
        :param index_partsToRanks: mappings of index partitions to ranks
        :param queries: [__cuda_array_interface__] of local query partitions
        :param query_m: number of total query rows
        :param query_partsToRanks: mappings of query partitions to ranks
        :return: 
        """

        if self._should_downcast:
            warnings.warn("Parameter should_downcast is deprecated, use "
                          "convert_dtype in fit and kneighbors "
                          " methods instead. ")
            convert_dtype = True

        self.__del__()

        self.n_dims = n

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        cdef vector[RankSizePair*] *index_vec = new vector[RankSizePair*]()

        cdef vector[RankSizePair*] *query_vec = new vector[RankSizePair*]()

        for idx, rankSize in enumerate(index_partsToRanks):
            rank, size = rankSize
            index = <RankSizePair*> malloc(sizeof(RankSizePair))
            index.rank = <int>rank
            index.size = <size_t>size

            index_vec.push_back(index)

        for idx, rankSize in enumerate(query_partsToRanks):
            rank, size = rankSize
            query = < RankSizePair * > malloc(sizeof(RankSizePair))
            query.rank = < int > rank
            query.size = < size_t > size

            query_vec.push_back(query)

        cdef vector[floatData_t*] *local_index_parts = \
                <vector[floatData_t*]*><size_t>self._build_dataFloat(indices)
        cdef vector[floatData_t*] *local_query_parts = \
                <vector[floatData_t*]*><size_t>self._build_dataFloat(queries)

        cdef PartDescriptor *index_descriptor = new PartDescriptor( \
            <size_t>index_m,
            <size_t>n,
            <vector[RankSizePair*]>deref(index_vec),
            <int>rank
        )
        cdef PartDescriptor *query_descriptor = new PartDescriptor( \
            <size_t>index_m,
            <size_t>n,
            <vector[RankSizePair*]>deref(index_vec),
            <int>rank)

        cdef uintptr_t X_ctype = -1
        cdef uintptr_t dev_ptr = -1

        return self


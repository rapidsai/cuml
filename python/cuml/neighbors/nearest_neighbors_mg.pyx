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

from libcpp cimport bool
from libcpp.memory cimport shared_ptr

import rmm
from libc.stdlib cimport malloc, free

from libc.stdint cimport uintptr_t, int64_t
from libc.stdlib cimport calloc, malloc, free

from numba import cuda
import rmm

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
        cdef floatData_t ** dataF = < floatData_t ** > \
                                      malloc(sizeof(floatData_t *) \
                                             * len(arr_interfaces))
        cdef uintptr_t input_ptr
        for x_i in range(len(arr_interfaces)):
            x = arr_interfaces[x_i]
            input_ptr = x["data"]
            print("Shape: " + str(x["shape"]))
            dataF[x_i] = < floatData_t * > malloc(sizeof(floatData_t))
            dataF[x_i].ptr = < float * > input_ptr
            dataF[x_i].totalSize = < size_t > (x["shape"][0] * x["shape"][1] * sizeof(float))
            print("Size: " + str((x["shape"][0] * x["shape"][1] * sizeof(float))))

        return <size_t>dataF

    def _freeFloatD(self, data, arr_interfaces):
        cdef uintptr_t data_ptr = data
        cdef floatData_t **d = <floatData_t**>data_ptr
        for x_i in range(len(arr_interfaces)):
            free(d[x_i])
        free(d)

    def kneighbors(self, indices, index_m, n, index_partsToRanks,
                         queries, query_m, query_partsToRanks):
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

        if len(X.shape) != 2:
            raise ValueError("data should be two dimensional")

        self.n_dims = X.shape[1]

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        index_rankSizePair = []
        query_rankSizePair = []

        for idx, rankSize in enumerate(index_partsToRanks):
            rank, size = rankSize
            index_rankSizePair[idx] = <RankSizePair*> malloc(sizeof(RankSizePair))
            index_rankSizePair[idx].rank = <int>rank
            index_rankSizePair[idx].size = <size_t>size

        for idx, rankSize in enumerate(query_partsToRanks):
            rank, size = rankSize
            query_rankSizePair[idx] = < RankSizePair * > malloc(sizeof(RankSizePair))
            query_rankSizePair[idx].rank = < int > rank
            query_rankSizePair[idx].size = < size_t > size

        cdef local_index_parts = <floatData_t**>self._build_dataFloat(indices)
        cdef local_query_parts = <floatData_t**>self._build_dataFloat(queries)


        cdef uintptr_t X_ctype = -1
        cdef uintptr_t dev_ptr = -1

        self.X_m, X_ctype, n_rows, n_cols, dtype = \
            input_to_dev_array(X, order='C', check_dtype=np.float32,
                               convert_to_dtype=(np.float32
                                                 if convert_dtype
                                                 else None))
        return self


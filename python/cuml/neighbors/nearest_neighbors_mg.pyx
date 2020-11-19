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

# distutils: language = c++

from cuml.neighbors import NearestNeighbors

import numpy as np
import pandas as pd
import cudf
import ctypes
import warnings
import typing

from cuml.common.base import Base
from cuml.common.array import CumlArray
from cuml.common import input_to_cuml_array
import cuml.internals

from cython.operator cimport dereference as deref

from cuml.raft.common.handle cimport handle_t
from cuml.common.opg_data_utils_mg import _build_part_inputs
import cuml.common.logger as logger

from libcpp cimport bool
from libcpp.memory cimport shared_ptr

import rmm
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector

from libc.stdint cimport uintptr_t, int64_t
from libc.stdlib cimport calloc, malloc, free

import rmm

cimport cuml.common.cuda


cdef extern from "cumlprims/opg/matrix/data.hpp" namespace \
        "MLCommon::Matrix":

    cdef cppclass Data[T]:
        Data(T *ptr, size_t totalSize)

    cdef cppclass floatData_t:
        floatData_t(float *ptr, size_t totalSize)
        float *ptr
        size_t totalSize

ctypedef Data[int64_t] int64Data_t


cdef extern from "cumlprims/opg/matrix/part_descriptor.hpp" namespace \
        "MLCommon::Matrix":

    cdef cppclass RankSizePair:
        int rank
        size_t size

    cdef cppclass PartDescriptor:
        PartDescriptor(size_t M,
                       size_t N,
                       vector[RankSizePair*] &partsToRanks,
                       int myrank)

cdef extern from "cuml/neighbors/knn_mg.hpp" namespace \
        "ML::KNN::opg":

    cdef void brute_force_knn(
        handle_t &handle,
        vector[int64Data_t*] &out_I,
        vector[floatData_t*] &out_D,
        vector[floatData_t*] &idx_data,
        PartDescriptor &idx_desc,
        vector[floatData_t*] &query_data,
        PartDescriptor &query_desc,
        bool rowMajorIndex,
        bool rowMajorQuery,
        int k,
        size_t batch_size,
        bool verbose
    ) except +


def _free_mem(index_desc, query_desc,
              out_i_vec, out_d_vec,
              local_index_parts,
              local_query_parts):
    cdef PartDescriptor *index_desc_c \
        = <PartDescriptor*><size_t>index_desc
    free(index_desc_c)

    cdef PartDescriptor *query_desc_c \
        = <PartDescriptor*><size_t>query_desc
    free(query_desc_c)

    cdef vector[int64Data_t *] *out_i_vec_c \
        = <vector[int64Data_t *]*><size_t>out_i_vec
    cdef int64Data_t *del_idx_ptr
    for elm in range(out_i_vec_c.size()):
        del_idx_ptr = out_i_vec_c.at(elm)
        del del_idx_ptr
    free(out_i_vec_c)

    cdef vector[floatData_t *] *out_d_vec_c \
        = <vector[floatData_t *]*><size_t>out_d_vec
    cdef floatData_t *del_ptr
    for elm in range(out_d_vec_c.size()):
        del_ptr = out_d_vec_c.at(elm)
        del del_ptr
    free(out_d_vec_c)

    cdef vector[RankSizePair *] *local_index_parts_c \
        = <vector[RankSizePair *]*><size_t>local_index_parts
    for elm in range(local_index_parts_c.size()):
        free(local_index_parts_c.at(elm))
    free(local_index_parts_c)

    cdef vector[RankSizePair *] *local_query_parts_c \
        = <vector[RankSizePair *]*><size_t>local_query_parts
    for elm in range(local_query_parts_c.size()):
        free(local_query_parts_c.at(elm))
    free(local_query_parts_c)


class NearestNeighborsMG(NearestNeighbors):
    """
    Multi-node multi-GPU Nearest Neighbors kneighbors query.

    NOTE: This implementation of NearestNeighbors is meant to be
    used with an initialized cumlCommunicator instance inside an
    existing distributed system. Refer to the Dask NearestNeighbors
     implementation in `cuml.dask.neighbors.nearest_neighbors`.

    The end-user API for multi-node multi-GPU NearestNeighbors is
    `cuml.dask.neighbors.NearestNeighbors`
    """
    def __init__(self, batch_size=1 << 21, **kwargs):
        super(NearestNeighborsMG, self).__init__(**kwargs)
        self.batch_size = batch_size

    @cuml.internals.api_base_return_generic_skipall
    def kneighbors(
        self,
        indices,
        index_m,
        n,
        index_parts_to_ranks,
        queries,
        query_m,
        query_parts_to_ranks,
        rank,
        n_neighbors=None,
        convert_dtype=True
    ) -> typing.Tuple[typing.List[CumlArray], typing.List[CumlArray]]:
        """
        Query the kneighbors of an index

        Parameters
        ----------
        indices: [__cuda_array_interface__] of local index partitions
        index_m: number of total index rows
        n: number of columns
        index_partsToRanks: mappings of index partitions to ranks
        queries: [__cuda_array_interface__] of local query partitions
        query_m: number of total query rows
        query_partsToRanks: mappings of query partitions to ranks
        rank: int rank of current worker
        n_neighbors: int number of nearest neighbors to query
        convert_dtype: since only float32 inputs are supported, should
               the input be automatically converted?

        Returns
        -------
        output indices, output distances
        """
        self._set_base_attributes(output_type=indices[0])

        # Specify the output return type
        cuml.internals.set_api_output_type(self._get_output_type(queries[0]))

        n_neighbors = self.n_neighbors if n_neighbors is None else n_neighbors

        self.n_dims = n

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        idx_cai, idx_local_parts, idx_desc = \
            _build_part_inputs(indices, index_parts_to_ranks,
                               index_m, n, rank, convert_dtype)

        q_cai, q_local_parts, q_desc = \
            _build_part_inputs(queries, query_parts_to_ranks,
                               query_m, n, rank, convert_dtype)

        cdef vector[int64Data_t*] *out_i_vec \
            = new vector[int64Data_t*]()
        cdef vector[floatData_t*] *out_d_vec \
            = new vector[floatData_t*]()

        output_i_arrs = []
        output_d_arrs = []

        cdef uintptr_t i_ptr
        cdef uintptr_t d_ptr

        for query_part in q_cai:

            n_rows = query_part.shape[0]
            i_ary = CumlArray.zeros((n_rows, n_neighbors),
                                    order="C",
                                    dtype=np.int64)
            d_ary = CumlArray.zeros((n_rows, n_neighbors),
                                    order="C",
                                    dtype=np.float32)

            output_i_arrs.append(i_ary)
            output_d_arrs.append(d_ary)

            i_ptr = i_ary.ptr
            d_ptr = d_ary.ptr

            out_i_vec.push_back(new int64Data_t(
                <int64_t*>i_ptr, n_rows * n_neighbors))

            out_d_vec.push_back(new floatData_t(
                <float*>d_ptr, n_rows * n_neighbors))

        is_verbose = logger.should_log_for(logger.level_debug)
        brute_force_knn(
            handle_[0],
            deref(out_i_vec),
            deref(out_d_vec),
            deref(<vector[floatData_t*]*><uintptr_t>idx_local_parts),
            deref(<PartDescriptor*><uintptr_t>idx_desc),
            deref(<vector[floatData_t*]*><uintptr_t>q_local_parts),
            deref(<PartDescriptor*><uintptr_t>q_desc),
            False,  # column-major index
            False,  # column-major query
            n_neighbors,
            <size_t>self.batch_size,
            <bool>is_verbose
        )

        self.handle.sync()

        _free_mem(<size_t>idx_desc,
                  <size_t>q_desc,
                  <size_t>out_i_vec,
                  <size_t>out_d_vec,
                  <size_t>idx_local_parts,
                  <size_t>q_local_parts)

        return output_i_arrs, output_d_arrs

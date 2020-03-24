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
from libcpp.vector cimport vector

from libc.stdint cimport uintptr_t, int64_t
from libc.stdlib cimport calloc, malloc, free

import rmm


cimport cuml.common.handle
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

cdef extern from "cumlprims/opg/selection/knn.hpp" namespace \
        "MLCommon::Selection::opg":

    cdef void brute_force_knn(
        cumlHandle &handle,
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


def _free_float_d(data):
    cdef uintptr_t data_ptr = <size_t>data
    cdef vector[floatData_t*] *d = <vector[floatData_t*]*>data_ptr
    for x_i in range(d.size()):
        free(d.at(x_i))
    free(d)


def _build_float_d(arr_interfaces):
    """
    Instantiate a container object for a float data pointer
    and size.

    Parameters
    ----------
    arr_interfaces:
    """
    cdef vector[floatData_t *] * dataF = new vector[floatData_t *]()

    cdef uintptr_t input_ptr
    for x_i in range(len(arr_interfaces)):
        x = arr_interfaces[x_i]
        input_ptr = x["data"]
        data = <floatData_t *> malloc(sizeof(floatData_t))
        data.ptr = < float * > input_ptr
        data.totalSize = <size_t> (x["shape"][0] *
                                   x["shape"][1] *
                                   sizeof(float))

        dataF.push_back(data)

    return < size_t > dataF


def _free_mem(index_vec, index_desc,
              query_vec, query_desc,
              out_i_vec, out_d_vec,
              local_index_parts,
              local_query_parts):

    cdef vector[floatData_t *] *index_vec_c \
        = <vector[floatData_t *]*><size_t>index_vec
    cdef PartDescriptor *index_desc_c \
        = <PartDescriptor*><size_t>index_desc

    _free_float_d(<size_t>index_vec_c)
    free(index_desc_c)

    cdef vector[floatData_t *] *query_vec_c \
        = <vector[floatData_t *]*><size_t>query_vec
    cdef PartDescriptor *query_desc_c \
        = <PartDescriptor*><size_t>query_desc
    _free_float_d(<size_t>query_vec_c)
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


def _build_part_inputs(cuda_arr_ifaces,
                       parts_to_ranks,
                       m, n, local_rank,
                       convert_dtype):

    cdef vector[RankSizePair*] *vec = new vector[RankSizePair*]()

    arr_ints = []
    for arr in cuda_arr_ifaces:
        X_m, input_ptr, n_rows, n_cols, dtype = \
            input_to_dev_array(arr, order="F",
                               convert_to_dtype=(np.float32
                                                 if convert_dtype
                                                 else None),
                               check_dtype=[np.float32])
        arr_ints.append({"obj": X_m,
                         "data": input_ptr,
                         "shape": (n_rows, n_cols)})

    for rankSize in parts_to_ranks:
        rank, size = rankSize
        rsp = <RankSizePair*> malloc(sizeof(RankSizePair))
        rsp.rank = <int>rank
        rsp.size = <size_t>size

        vec.push_back(rsp)

    cdef vector[floatData_t*] *local_parts \
        = <vector[floatData_t*]*><size_t> _build_float_d(arr_ints)

    cdef PartDescriptor *descriptor \
        = new PartDescriptor(<size_t>m,
                             <size_t>n,
                             <vector[RankSizePair*]>deref(vec),
                             <int>local_rank)

    cdef uintptr_t rsp_ptr = <uintptr_t>vec
    cdef uintptr_t local_parts_ptr = <uintptr_t>local_parts
    cdef uintptr_t desc_ptr = <uintptr_t>descriptor

    return arr_ints, rsp_ptr, local_parts_ptr, desc_ptr


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
    def __init__(self, batch_size=1<<21, **kwargs):
        super(NearestNeighborsMG, self).__init__(**kwargs)
        self.batch_size = batch_size

    def kneighbors(self, indices, index_m, n, index_parts_to_ranks,
                   queries, query_m, query_parts_to_ranks,
                   rank, n_neighbors=None, convert_dtype=True):
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

        n_neighbors = self.n_neighbors if n_neighbors is None else n_neighbors

        self.n_dims = n

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        idx_cai, idx_rsp, idx_local_parts, idx_desc = \
            _build_part_inputs(indices, index_parts_to_ranks,
                               index_m, n, rank, convert_dtype)

        q_cai, q_rsp, q_local_parts, q_desc = \
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

            n_rows = query_part["shape"][0]
            i_ary = rmm.to_device(zeros((n_rows, n_neighbors),
                                        order="C",
                                        dtype=np.int64))
            d_ary = rmm.to_device(zeros((n_rows, n_neighbors),
                                        order="C",
                                        dtype=np.float32))

            output_i_arrs.append(i_ary)
            output_d_arrs.append(d_ary)

            i_ptr = get_dev_array_ptr(i_ary)
            d_ptr = get_dev_array_ptr(d_ary)

            out_i_vec.push_back(new int64Data_t(
                <int64_t*>i_ptr, n_rows * n_neighbors))

            out_d_vec.push_back(new floatData_t(
                <float*>d_ptr, n_rows * n_neighbors))

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
            <bool>self.verbose
        )

        self.handle.sync()

        output_i = list(map(lambda x: cudf.DataFrame.from_gpu_matrix(x),
                            output_i_arrs))
        output_d = list(map(lambda x: cudf.DataFrame.from_gpu_matrix(x),
                            output_d_arrs))

        _free_mem(<size_t>idx_rsp,
                  <size_t>idx_desc,
                  <size_t>q_rsp,
                  <size_t>q_desc,
                  <size_t>out_i_vec,
                  <size_t>out_d_vec,
                  <size_t>idx_local_parts,
                  <size_t>q_local_parts)

        return output_i, output_d

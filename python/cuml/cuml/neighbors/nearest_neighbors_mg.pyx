#
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

import typing

import cuml.internals.logger as logger
from cuml.common import input_to_cuml_array
from cuml.internals import api_base_return_generic_skipall
from cuml.internals.array import CumlArray
from cuml.neighbors import NearestNeighbors

from pylibraft.common.handle cimport handle_t

from cuml.common.opg_data_utils_mg cimport *

from cuml.common.opg_data_utils_mg import _build_part_inputs

from cython.operator cimport dereference as deref
from libc.stdint cimport uintptr_t
from libc.stdlib cimport free
from libcpp cimport bool
from libcpp.vector cimport vector


cdef extern from "cuml/neighbors/knn_mg.hpp" namespace "ML::KNN::opg" nogil:

    cdef void knn(
        handle_t &handle,
        vector[int64Data_t*] *out_I,
        vector[floatData_t*] *out_D,
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
    def __init__(self, *, batch_size=2000000, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    @api_base_return_generic_skipall
    def kneighbors(
        self,
        index,
        index_parts_to_ranks,
        index_nrows,
        query,
        query_parts_to_ranks,
        query_nrows,
        ncols,
        rank,
        n_neighbors,
        convert_dtype
    ) -> typing.Tuple[typing.List[CumlArray], typing.List[CumlArray]]:
        """
        Query the kneighbors of an index

        Parameters
        ----------
        index: [__cuda_array_interface__] of local index partitions
        index_parts_to_ranks: mappings of index partitions to ranks
        index_nrows: number of index rows
        query: [__cuda_array_interface__] of local query partitions
        query_parts_to_ranks: mappings of query partitions to ranks
        query_nrows: number of query rows
        ncols: number of columns
        rank: rank of current worker
        n_neighbors: number of nearest neighbors to query
        convert_dtype: since only float32 inputs are supported, should
               the input be automatically converted?

        Returns
        -------
        predictions : indices and distances
        """
        # Detect type
        self.get_out_type(index, query)

        self.n_neighbors = self.n_neighbors if n_neighbors is None \
            else n_neighbors

        # Build input arrays and descriptors for native code interfacing
        input = type(self).gen_local_input(
            index, index_parts_to_ranks, index_nrows, query,
            query_parts_to_ranks, query_nrows, ncols, rank, convert_dtype)

        query_cais = input['cais']['query']
        local_query_rows = list(map(lambda x: x.shape[0], query_cais))

        # Build indices and distances outputs for native code interfacing
        result = type(self).alloc_local_output(local_query_rows, self.n_neighbors)

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        is_verbose = logger.should_log_for(logger.level_enum.debug)

        # Launch distributed operations
        knn(
            handle_[0],
            <vector[int64Data_t*]*><uintptr_t>result['indices'],
            <vector[floatData_t*]*><uintptr_t>result['distances'],
            deref(<vector[floatData_t*]*><uintptr_t>
                  input['index']['local_parts']),
            deref(<PartDescriptor*><uintptr_t>input['index']['desc']),
            deref(<vector[floatData_t*]*><uintptr_t>
                  input['query']['local_parts']),
            deref(<PartDescriptor*><uintptr_t>input['query']['desc']),
            <bool>False,  # column-major index
            <bool>False,  # column-major query
            <int>self.n_neighbors,
            <size_t>self.batch_size,
            <bool>is_verbose
        )
        self.handle.sync()

        # Release memory
        type(self).free_mem(input, result)

        return result['cais']['distances'], result['cais']['indices']

    def get_out_type(self, index, query):
        if len(index) > 0:
            self._set_base_attributes(output_type=index[0])
        if len(query) > 0:
            self._set_base_attributes(output_type=query[0])

    @staticmethod
    def gen_local_input(index, index_parts_to_ranks, index_nrows,
                        query, query_parts_to_ranks, query_nrows,
                        ncols, rank, convert_dtype):
        index_dask = [d[0] if isinstance(d, (list, tuple))
                      else d for d in index]

        index_cai, index_local_parts, index_desc = \
            _build_part_inputs(index_dask, index_parts_to_ranks, index_nrows,
                               ncols, rank, convert_dtype)

        query_cai, query_local_parts, query_desc = \
            _build_part_inputs(query, query_parts_to_ranks, query_nrows,
                               ncols, rank, convert_dtype)

        return {
            'index': {
                'local_parts': <uintptr_t>index_local_parts,
                'desc': <uintptr_t>index_desc
            },
            'query': {
                'local_parts': <uintptr_t>query_local_parts,
                'desc': <uintptr_t>query_desc
            },
            'cais': {
                'index': index_cai,
                'query': query_cai
            },
        }

    @staticmethod
    def gen_local_labels(index, convert_dtype, dtype):
        cdef vector[int_ptr_vector] *out_local_parts_i32
        cdef vector[float_ptr_vector] *out_local_parts_f32

        outputs = [d[1] for d in index]
        n_out = len(outputs)

        if dtype == 'int32':
            out_local_parts_i32 = new vector[int_ptr_vector](<int>n_out)
        elif dtype == 'float32':
            out_local_parts_f32 = new vector[float_ptr_vector](<int>n_out)
        else:
            raise ValueError('Wrong dtype')

        def to_cupy(data):
            data, _, _, _ = input_to_cuml_array(data)
            return data.to_output('cupy')

        outputs_cai = []
        for i, arr in enumerate(outputs):
            arr = to_cupy(arr)
            n_features = arr.shape[1] if arr.ndim != 1 else 1
            for j in range(n_features):
                col = arr[:, j] if n_features != 1 else arr
                out_ai, _, _, _ = \
                    input_to_cuml_array(col, order="F",
                                        convert_to_dtype=(dtype
                                                          if convert_dtype
                                                          else None),
                                        check_dtype=[dtype])
                outputs_cai.append(out_ai)
                if dtype == 'int32':
                    out_local_parts_i32.at(i).push_back(<int*><uintptr_t>
                                                        out_ai.ptr)
                else:
                    out_local_parts_f32.at(i).push_back(<float*><uintptr_t>
                                                        out_ai.ptr)

        return {
            'labels':
                <uintptr_t>out_local_parts_i32 if dtype == 'int32'
                else <uintptr_t>out_local_parts_f32,
            'cais': [outputs_cai]
        }

    @staticmethod
    def alloc_local_output(local_query_rows, n_neighbors):
        cdef vector[int64Data_t*] *indices_local_parts \
            = new vector[int64Data_t*]()
        cdef vector[floatData_t*] *dist_local_parts \
            = new vector[floatData_t*]()

        indices_cai = []
        dist_cai = []
        for n_rows in local_query_rows:
            i_cai = CumlArray.zeros(shape=(n_rows, n_neighbors),
                                    order="C", dtype='int64')
            d_cai = CumlArray.zeros(shape=(n_rows, n_neighbors),
                                    order="C", dtype='float32')

            indices_cai.append(i_cai)
            dist_cai.append(d_cai)

            indices_local_parts.push_back(new int64Data_t(
                <int64_t*><uintptr_t>i_cai.ptr, n_rows * n_neighbors))

            dist_local_parts.push_back(new floatData_t(
                <float*><uintptr_t>d_cai.ptr, n_rows * n_neighbors))

        return {
            'indices': <uintptr_t>indices_local_parts,
            'distances': <uintptr_t>dist_local_parts,
            'cais': {
                'indices': indices_cai,
                'distances': dist_cai
            }
        }

    @staticmethod
    def free_mem(input, result=None):
        cdef floatData_t *f_ptr
        cdef vector[floatData_t*] *f_lp

        for input_type in ['index', 'query']:
            ilp = input[input_type]['local_parts']
            f_lp = <vector[floatData_t *]*><uintptr_t>ilp
            for i in range(f_lp.size()):
                f_ptr = f_lp.at(i)
                free(<void*>f_ptr)
            free(<void*><uintptr_t>f_lp)

            free(<void*><uintptr_t>input[input_type]['desc'])

        cdef int64Data_t *i64_ptr
        cdef vector[int64Data_t*] *i64_lp

        if result:

            f_lp = <vector[floatData_t *]*><uintptr_t>result['distances']
            for i in range(f_lp.size()):
                f_ptr = f_lp.at(i)
                free(<void*>f_ptr)
            free(<void*><uintptr_t>f_lp)

            i64_lp = <vector[int64Data_t *]*><uintptr_t>result['indices']
            for i in range(i64_lp.size()):
                i64_ptr = i64_lp.at(i)
                free(<void*>i64_ptr)
            free(<void*><uintptr_t>i64_lp)

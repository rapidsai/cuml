#
# Copyright (c) 2020, NVIDIA CORPORATION.
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

import numpy as np

import cuml.internals
from cuml.common.array import CumlArray
from cuml.raft.common.handle cimport handle_t
from cuml.common import input_to_cuml_array
from cuml.common.opg_data_utils_mg cimport *
from cuml.common.opg_data_utils_mg import _build_part_inputs

import rmm
from libc.stdlib cimport calloc, malloc, free
from cython.operator cimport dereference as deref
from libc.stdint cimport uintptr_t
from libcpp cimport bool
from libcpp.memory cimport shared_ptr

from cuml.neighbors.nearest_neighbors_mg import NearestNeighbors
from cudf.core import DataFrame as cudfDataFrame


class KNeighborsMG(NearestNeighbors):
    def __init__(self, batch_size=1024, **kwargs):
        super(KNeighborsMG, self).__init__(**kwargs)
        self.batch_size = batch_size

    def get_out_type(self, data, query):
        if len(data) > 0:
            self._set_base_attributes(output_type=data[0])
        out_type = self.output_type
        if len(query) > 0:
            out_type = self._get_output_type(query[0])

        cuml.internals.set_api_output_type(out_type)
        return out_type

    def gen_local_input(self, data, data_parts_to_ranks, data_nrows,
                        query, query_parts_to_ranks, query_nrows,
                        ncols, rank, convert_dtype):
        data_dask = [d[0] for d in data]
        self.n_dims = ncols

        data_cai, data_local_parts, data_desc = \
            _build_part_inputs(data_dask, data_parts_to_ranks, data_nrows,
                               ncols, rank, convert_dtype)

        query_cai, query_local_parts, query_desc = \
            _build_part_inputs(query, query_parts_to_ranks, query_nrows,
                               ncols, rank, convert_dtype)

        return {
            'data': {
                'local_parts': <uintptr_t>data_local_parts,
                'desc': <uintptr_t>data_desc
            },
            'query': {
                'local_parts': <uintptr_t>query_local_parts,
                'desc': <uintptr_t>query_desc
            },
            'cais': {
                'data': data_cai,
                'query': query_cai
            },
        }

    def gen_local_output(self, data, convert_dtype, dtype):
        cdef vector[int_ptr_vector] *out_local_parts_i32
        cdef vector[float_ptr_vector] *out_local_parts_f32

        outputs = [d[1] for d in data]
        n_out = len(outputs)

        if dtype == 'int32':
            out_local_parts_i32 = new vector[int_ptr_vector](<int>n_out)
        elif dtype == 'float32':
            out_local_parts_f32 = new vector[float_ptr_vector](<int>n_out)
        else:
            raise ValueError('Wrong dtype')

        outputs_cai = []
        for i, arr in enumerate(outputs):
            for j in range(arr.shape[1]):
                if isinstance(arr, cudfDataFrame):
                    col = arr.iloc[:, j]
                else:
                    col = arr[:, j]
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
            'outputs':
                <uintptr_t>out_local_parts_i32 if dtype == 'int32'
                else <uintptr_t>out_local_parts_f32,
            'cais': [outputs_cai]
        }

    def alloc_local_output(self, local_query_rows):
        cdef vector[int64Data_t*] *indices_local_parts \
            = new vector[int64Data_t*]()
        cdef vector[floatData_t*] *dist_local_parts \
            = new vector[floatData_t*]()

        indices_cai = []
        dist_cai = []
        for n_rows in local_query_rows:
            i_cai = CumlArray.zeros(shape=(n_rows, self.n_neighbors),
                                    order="C", dtype=np.int64)
            d_cai = CumlArray.zeros(shape=(n_rows, self.n_neighbors),
                                    order="C", dtype=np.float32)

            indices_cai.append(i_cai)
            dist_cai.append(d_cai)

            indices_local_parts.push_back(new int64Data_t(
                <int64_t*><uintptr_t>i_cai.ptr, n_rows * self.n_neighbors))

            dist_local_parts.push_back(new floatData_t(
                <float*><uintptr_t>d_cai.ptr, n_rows * self.n_neighbors))

        return {
            'indices': <uintptr_t>indices_local_parts,
            'distances': <uintptr_t>dist_local_parts,
            'cais': {
                'indices': indices_cai,
                'distances': dist_cai
            }
        }

    def free_mem(self, input, result=None):
        cdef floatData_t *f_ptr
        cdef vector[floatData_t*] *f_lp

        for input_type in ['data', 'query']:
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

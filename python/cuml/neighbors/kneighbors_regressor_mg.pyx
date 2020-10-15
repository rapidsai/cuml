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

from cuml.neighbors.kneighbors_mg import KNeighborsMG
from cudf.core import DataFrame as cudfDataFrame

cdef extern from "cuml/neighbors/knn_mg.hpp" namespace \
        "ML::KNN::opg":

    cdef void knn_regress(
        handle_t &handle,
        vector[floatData_t*] *out,
        vector[int64Data_t*] *out_I,
        vector[floatData_t*] *out_D,
        vector[floatData_t*] &idx_data,
        PartDescriptor &idx_desc,
        vector[floatData_t*] &query_data,
        PartDescriptor &query_desc,
        vector[float_ptr_vector] &y,
        bool rowMajorIndex,
        bool rowMajorQuery,
        int k,
        int n_outputs,
        size_t batch_size,
        bool verbose
    ) except +


class KNeighborsRegressorMG(KNeighborsMG):
    """
    Multi-node Multi-GPU K-Nearest Neighbors Regressor Model.

    K-Nearest Neighbors Regressor is an instance-based learning technique,
    that keeps training samples around for prediction, rather than trying
    to learn a generalizable set of model parameters.
    """
    def __init__(self, batch_size=1024, **kwargs):
        super(KNeighborsRegressorMG, self).__init__(**kwargs)
        self.batch_size = batch_size

    def predict(self, data, data_parts_to_ranks, data_nrows,
                query, query_parts_to_ranks, query_nrows,
                ncols, n_outputs, rank, convert_dtype):
        """
        Predict outputs for a query from previously stored index
        and index labels.
        The process is done in a multi-node multi-GPU fashion.

        Parameters
        ----------
        data: [__cuda_array_interface__] of local index and labels partitions
        data_parts_to_ranks: mappings of data partitions to ranks
        data_nrows: number of total data rows
        query: [__cuda_array_interface__] of local query partitions
        query_parts_to_ranks: mappings of query partitions to ranks
        query_nrows: number of total query rows
        ncols: number of columns
        rank: int rank of current worker
        convert_dtype: since only float32 inputs are supported, should
               the input be automatically converted?

        Returns
        -------
        predictions : outputs, indices, distances
        """
        out_type = self.get_out_type(data, query)

        out_type = self.get_out_type(data, query)

        input = self.gen_local_input(data, data_parts_to_ranks, data_nrows,
                                     query, query_parts_to_ranks, query_nrows,
                                     ncols, rank, convert_dtype)

        output = self.gen_local_output(data, convert_dtype, dtype='float32')

        query_cais = input['cais']['query']
        local_query_rows = list(map(lambda x: x.shape[0], query_cais))
        result = self.alloc_local_output(local_query_rows)

        cdef vector[floatData_t*] *out_result_local_parts \
            = new vector[floatData_t*]()
        output_cais = []
        for n_rows in local_query_rows:
            o_cai = CumlArray.zeros(shape=(n_rows, n_outputs),
                                    order="C", dtype=np.float32)
            output_cais.append(o_cai)
            out_result_local_parts.push_back(new floatData_t(
                <float*><uintptr_t>o_cai.ptr, n_rows * n_outputs))

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        knn_regress(
            handle_[0],
            out_result_local_parts,
            <vector[int64Data_t*]*><uintptr_t>result['indices'],
            <vector[floatData_t*]*><uintptr_t>result['distances'],
            deref(<vector[floatData_t*]*><uintptr_t>
                  input['data']['local_parts']),
            deref(<PartDescriptor*><uintptr_t>input['data']['desc']),
            deref(<vector[floatData_t*]*><uintptr_t>
                  input['query']['local_parts']),
            deref(<PartDescriptor*><uintptr_t>input['query']['desc']),
            deref(<vector[float_ptr_vector]*><uintptr_t>output['outputs']),
            <bool>False,  # column-major index
            <bool>False,  # column-major query
            <int>self.n_neighbors,
            <int>n_outputs,
            <size_t>self.batch_size,
            <bool>self.verbose
        )

        self.handle.sync()

        self.free_mem(input, result)
        free(<void*><uintptr_t>output['outputs'])

        for i in range(out_result_local_parts.size()):
            free(<void*>out_result_local_parts.at(i))
        free(<void*><uintptr_t>out_result_local_parts)

        output = list(map(lambda o: o.to_output(out_type), output_cais))
        output_i = list(map(lambda o: o.to_output(out_type),
                            result['cais']['indices']))
        output_d = list(map(lambda o: o.to_output(out_type),
                            result['cais']['distances']))

        return output, output_i, output_d

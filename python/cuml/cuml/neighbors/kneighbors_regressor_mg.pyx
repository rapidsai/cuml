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
from cuml.internals import api_base_return_generic_skipall
from cuml.internals.array import CumlArray
from cuml.neighbors.nearest_neighbors_mg import NearestNeighborsMG

from cython.operator cimport dereference as deref
from libc.stdint cimport uintptr_t
from libc.stdlib cimport free
from libcpp cimport bool
from libcpp.vector cimport vector
from pylibraft.common.handle cimport handle_t

from cuml.common.opg_data_utils_mg cimport *


cdef extern from "cuml/neighbors/knn_mg.hpp" namespace "ML::KNN::opg" nogil:

    cdef void knn_regress(
        handle_t &handle,
        vector[floatData_t*] *out,
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


class KNeighborsRegressorMG(NearestNeighborsMG):
    """
    Multi-node Multi-GPU K-Nearest Neighbors Regressor Model.

    K-Nearest Neighbors Regressor is an instance-based learning technique,
    that keeps training samples around for prediction, rather than trying
    to learn a generalizable set of model parameters.
    """
    def __init__(self, **kwargs):
        super(KNeighborsRegressorMG, self).__init__(**kwargs)

    @api_base_return_generic_skipall
    def predict(
        self,
        index,
        index_parts_to_ranks,
        index_nrows,
        query,
        query_parts_to_ranks,
        query_nrows,
        ncols,
        n_outputs,
        rank,
        convert_dtype
    ) -> typing.List[CumlArray]:
        """
        Predict outputs for a query from previously stored index
        and index labels.
        The process is done in a multi-node multi-GPU fashion.

        Parameters
        ----------
        index: [__cuda_array_interface__] of local index partitions
        index_parts_to_ranks: mappings of index partitions to ranks
        index_nrows: number of index rows
        query: [__cuda_array_interface__] of local query partitions
        query_parts_to_ranks: mappings of query partitions to ranks
        query_nrows: number of query rows
        ncols: number of columns
        n_outputs: number of outputs columns
        rank: rank of current worker
        convert_dtype: since only float32 inputs are supported, should
               the input be automatically converted?

        Returns
        -------
        predictions : labels
        """
        # Detect type
        self.get_out_type(index, query)

        # Build input arrays and descriptors for native code interfacing
        input = type(self).gen_local_input(
            index, index_parts_to_ranks, index_nrows, query,
            query_parts_to_ranks, query_nrows, ncols, rank, convert_dtype)

        # Build input labels arrays and descriptors for native code interfacing
        labels = type(self).gen_local_labels(index, convert_dtype, dtype='float32')

        query_cais = input['cais']['query']
        local_query_rows = list(map(lambda x: x.shape[0], query_cais))

        # Build labels output array for native code interfacing
        cdef vector[floatData_t*] *out_result_local_parts \
            = new vector[floatData_t*]()
        output_cais = []
        for n_rows in local_query_rows:
            o_cai = CumlArray.zeros(shape=(n_rows, n_outputs),
                                    order="C", dtype='float32')
            output_cais.append(o_cai)
            out_result_local_parts.push_back(new floatData_t(
                <float*><uintptr_t>o_cai.ptr, n_rows * n_outputs))

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        is_verbose = logger.should_log_for(logger.level_enum.debug)

        # Launch distributed operations
        knn_regress(
            handle_[0],
            out_result_local_parts,
            deref(<vector[floatData_t*]*><uintptr_t>
                  input['index']['local_parts']),
            deref(<PartDescriptor*><uintptr_t>input['index']['desc']),
            deref(<vector[floatData_t*]*><uintptr_t>
                  input['query']['local_parts']),
            deref(<PartDescriptor*><uintptr_t>input['query']['desc']),
            deref(<vector[float_ptr_vector]*><uintptr_t>labels['labels']),
            <bool>False,  # column-major index
            <bool>False,  # column-major query
            <int>self.n_neighbors,
            <int>n_outputs,
            <size_t>self.batch_size,
            <bool>is_verbose
        )
        self.handle.sync()

        # Release memory
        type(self).free_mem(input)
        free(<void*><uintptr_t>labels['labels'])
        for i in range(out_result_local_parts.size()):
            free(<void*>out_result_local_parts.at(i))
        free(<void*><uintptr_t>out_result_local_parts)

        return output_cais

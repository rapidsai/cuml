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

from pylibraft.common.handle cimport handle_t

from cuml.common.opg_data_utils_mg cimport *

from cuml.common import input_to_cuml_array

from cython.operator cimport dereference as deref
from libc.stdint cimport uintptr_t
from libc.stdlib cimport free
from libcpp cimport bool
from libcpp.vector cimport vector


cdef extern from "cuml/neighbors/knn_mg.hpp" namespace "ML::KNN::opg" nogil:

    cdef void knn_classify(
        handle_t &handle,
        vector[intData_t*] *out,
        vector[float_ptr_vector] *probas,
        vector[floatData_t*] &idx_data,
        PartDescriptor &idx_desc,
        vector[floatData_t*] &query_data,
        PartDescriptor &query_desc,
        vector[int_ptr_vector] &y,
        vector[int*] &uniq_labels,
        vector[int] &n_unique,
        bool rowMajorIndex,
        bool rowMajorQuery,
        bool probas_only,
        int k,
        size_t batch_size,
        bool verbose
    ) except +


class KNeighborsClassifierMG(NearestNeighborsMG):
    """
    Multi-node Multi-GPU K-Nearest Neighbors Classifier Model.

    K-Nearest Neighbors Classifier is an instance-based learning technique,
    that keeps training samples around for prediction, rather than trying
    to learn a generalizable set of model parameters.
    """
    def __init__(self, **kwargs):
        super(KNeighborsClassifierMG, self).__init__(**kwargs)

    @api_base_return_generic_skipall
    def predict(
        self,
        index,
        index_parts_to_ranks,
        index_nrows,
        query,
        query_parts_to_ranks,
        query_nrows,
        uniq_labels,
        n_unique,
        ncols,
        rank,
        convert_dtype
    ) -> typing.List[CumlArray]:
        """
        Predict labels for a query from previously stored index
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
        uniq_labels: array of arrays of possible labels for columns
        n_unique: array with number of possible labels for each columns
        ncols: number of columns
        rank: rank of current worker
        n_neighbors: number of nearest neighbors to query
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
        labels = type(self).gen_local_labels(index, convert_dtype, 'int32')

        query_cais = input['cais']['query']
        local_query_rows = list(map(lambda x: x.shape[0], query_cais))

        # Build uniq_labels_vec vector for native code interfacing
        uniq_labels_d, _, _, _ = \
            input_to_cuml_array(uniq_labels, order='C', check_dtype='int32',
                                convert_to_dtype='int32')
        cdef int* ptr = <int*><uintptr_t>uniq_labels_d.ptr
        cdef vector[int*] *uniq_labels_vec = new vector[int*]()
        for i in range(uniq_labels_d.shape[0]):
            uniq_labels_vec.push_back(<int*>ptr)
            ptr += <int>uniq_labels_d.shape[1]

        # Build n_unique_vec vector for native code interfacing
        cdef vector[int] *n_unique_vec = \
            new vector[int]()
        for uniq_label in n_unique:
            n_unique_vec.push_back(uniq_label)

        n_outputs = len(n_unique)

        # Build labels output array for native code interfacing
        cdef vector[intData_t*] *out_result_local_parts \
            = new vector[intData_t*]()
        output_cais = []
        for n_rows in local_query_rows:
            o_cai = CumlArray.zeros(shape=(n_rows, n_outputs),
                                    order="C", dtype='int32')
            output_cais.append(o_cai)
            out_result_local_parts.push_back(new intData_t(
                <int*><uintptr_t>o_cai.ptr, n_rows * n_outputs))

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        is_verbose = logger.should_log_for(logger.level_enum.debug)
        knn_classify(
            handle_[0],
            out_result_local_parts,
            <vector[float_ptr_vector]*>0,
            deref(<vector[floatData_t*]*><uintptr_t>
                  input['index']['local_parts']),
            deref(<PartDescriptor*><uintptr_t>input['index']['desc']),
            deref(<vector[floatData_t*]*><uintptr_t>
                  input['query']['local_parts']),
            deref(<PartDescriptor*><uintptr_t>input['query']['desc']),
            deref(<vector[int_ptr_vector]*><uintptr_t>labels['labels']),
            deref(<vector[int*]*><uintptr_t>uniq_labels_vec),
            deref(<vector[int]*><uintptr_t>n_unique_vec),
            <bool>False,  # column-major index
            <bool>False,  # column-major query
            <bool>False,
            <int>self.n_neighbors,
            <size_t>self.batch_size,
            <bool>is_verbose
        )

        self.handle.sync()

        # Release memory
        type(self).free_mem(input)
        free(<void*><uintptr_t>labels['labels'])
        type(self)._free_unique(
            <uintptr_t>uniq_labels_vec, <uintptr_t>n_unique_vec)
        for i in range(out_result_local_parts.size()):
            free(<void*>out_result_local_parts.at(i))
        free(<void*><uintptr_t>out_result_local_parts)

        return output_cais

    @api_base_return_generic_skipall
    def predict_proba(self, index, index_parts_to_ranks, index_nrows,
                      query, query_parts_to_ranks, query_nrows,
                      uniq_labels, n_unique, ncols, rank,
                      convert_dtype) -> tuple:
        """
        Predict labels for a query from previously stored index
        and index labels.
        The process is done in a multi-node multi-GPU fashion.

        Parameters
        ----------
        index: [__cuda_array_interface__] of local index and labels partitions
        index_parts_to_ranks: mappings of index partitions to ranks
        index_nrows: number of total index rows
        query: [__cuda_array_interface__] of local query partitions
        query_parts_to_ranks: mappings of query partitions to ranks
        query_nrows: number of total query rows
        uniq_labels: array of labels of a column
        n_unique: array with number of possible labels for each columns
        ncols: number of columns
        rank: int rank of current worker
        convert_dtype: since only float32 inputs are supported, should
               the input be automatically converted?

        Returns
        -------
        predictions : labels, indices, distances
        """
        # Detect type
        self.get_out_type(index, query)

        # Build input arrays and descriptors for native code interfacing
        input = type(self).gen_local_input(
            index, index_parts_to_ranks, index_nrows, query,
            query_parts_to_ranks, query_nrows, ncols, rank, convert_dtype)

        # Build input labels arrays and descriptors for native code interfacing
        labels = type(self).gen_local_labels(index, convert_dtype, dtype='int32')

        # Build uniq_labels_vec vector for native code interfacing
        uniq_labels_d, _, _, _ = \
            input_to_cuml_array(uniq_labels, order='C', check_dtype='int32',
                                convert_to_dtype='int32')
        cdef int* ptr = <int*><uintptr_t>uniq_labels_d.ptr
        cdef vector[int*] *uniq_labels_vec = new vector[int*]()
        for i in range(uniq_labels_d.shape[0]):
            uniq_labels_vec.push_back(<int*>ptr)
            ptr += <int>uniq_labels_d.shape[1]

        # Build n_unique_vec vector for native code interfacing
        cdef vector[int] *n_unique_vec = \
            new vector[int]()
        for uniq_label in n_unique:
            n_unique_vec.push_back(uniq_label)

        query_cais = input['cais']['query']
        local_query_rows = list(map(lambda x: x.shape[0], query_cais))
        n_local_queries = len(local_query_rows)

        cdef vector[float_ptr_vector] *probas_local_parts \
            = new vector[float_ptr_vector](n_local_queries)

        n_outputs = len(n_unique)

        # Build probas output array for native code interfacing
        proba_cais = [[] for i in range(n_outputs)]
        for query_idx, n_rows in enumerate(local_query_rows):
            for target_idx, n_classes in enumerate(n_unique):
                p_cai = CumlArray.zeros(shape=(n_rows, n_classes),
                                        order="C", dtype='float32')
                proba_cais[target_idx].append(p_cai)

                probas_local_parts.at(query_idx).push_back(<float*><uintptr_t>
                                                           p_cai.ptr)

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        is_verbose = logger.should_log_for(logger.level_enum.debug)

        # Launch distributed operations
        knn_classify(
            handle_[0],
            <vector[intData_t*]*>0,
            probas_local_parts,
            deref(<vector[floatData_t*]*><uintptr_t>
                  input['index']['local_parts']),
            deref(<PartDescriptor*><uintptr_t>input['index']['desc']),
            deref(<vector[floatData_t*]*><uintptr_t>
                  input['query']['local_parts']),
            deref(<PartDescriptor*><uintptr_t>input['query']['desc']),
            deref(<vector[int_ptr_vector]*><uintptr_t>labels['labels']),
            deref(<vector[int*]*><uintptr_t>uniq_labels_vec),
            deref(<vector[int]*><uintptr_t>n_unique_vec),
            <bool>False,  # column-major index
            <bool>False,  # column-major query
            <bool>True,
            <int>self.n_neighbors,
            <size_t>self.batch_size,
            <bool>is_verbose
        )
        self.handle.sync()

        # Release memory
        type(self).free_mem(input)
        free(<void*><uintptr_t>labels['labels'])
        type(self)._free_unique(
            <uintptr_t>uniq_labels_vec, <uintptr_t>n_unique_vec)
        free(<void*><uintptr_t>probas_local_parts)

        return tuple(proba_cais)

    @staticmethod
    def _free_unique(uniq_labels, n_unique):
        free(<void*><uintptr_t>uniq_labels)
        free(<void*><uintptr_t>n_unique)

#
# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

import cuml.common.opg_data_utils_mg as opg
import cuml.internals
from cuml.common.sparse_utils import is_sparse
from cuml.internals.array import CumlArray
from cuml.internals.array_sparse import SparseCumlArray
from cuml.internals.input_utils import input_to_cuml_array

from libc.stdint cimport uintptr_t

from cuml.common.opg_data_utils_mg cimport *


class MGFitMixin(object):

    @cuml.internals.api_base_return_any_skipall
    def fit(
        self,
        input_data,
        n_rows,
        n_cols,
        partsToSizes,
        rank,
        order='F',
        convert_index=np.int32,
    ):
        """
        Fit function for MNMG linear regression classes
        This not meant to be used as
        part of the public API.
        :param X: array of local dataframes / array partitions
        :param n_rows: total number of rows
        :param n_cols: total number of cols
        :param partsToSizes: array of tuples in the format: [(rank,size)]
        :return: self
        """

        self._set_output_type(input_data[0][0])
        self._set_n_features_in(n_cols)
        sparse_input = is_sparse(input_data[0][0])

        X_arys = []
        y_arys = []

        for i in range(len(input_data)):
            if i == 0:
                check_dtype = [np.float32, np.float64]
            else:
                check_dtype = self.dtype

            if sparse_input:
                X_m = SparseCumlArray(input_data[i][0], convert_index=convert_index)
                _, self.n_cols = X_m.shape
            else:
                X_m, _, self.n_cols, _ = \
                    input_to_cuml_array(input_data[i][0], check_dtype=check_dtype, order=order)

            X_arys.append(X_m)

            if i == 0:
                self.dtype = X_m.dtype
                if sparse_input:
                    self.index_dtype = X_m.indptr.dtype

            y_m, *_ = input_to_cuml_array(input_data[i][1],
                                          check_dtype=self.dtype)
            y_arys.append(y_m)

        self.coef_ = CumlArray.zeros(self.n_cols,
                                     dtype=self.dtype)
        cdef uintptr_t coef_ptr = self.coef_.ptr
        coef_ptr_arg = <size_t>coef_ptr

        cdef uintptr_t rank_to_sizes = opg.build_rank_size_pair(partsToSizes,
                                                                rank)

        cdef uintptr_t part_desc = opg.build_part_descriptor(n_rows,
                                                             n_cols,
                                                             rank_to_sizes,
                                                             rank)

        cdef uintptr_t X_arg
        cdef uintptr_t y_arg = opg.build_data_t(y_arys)

        cdef uintptr_t X_cols
        cdef uintptr_t X_row_ids

        if sparse_input is False:

            X_arg = opg.build_data_t(X_arys)

            # call inheriting class _fit that does all cython pointers and calls
            self._fit(X=X_arg,
                      y=y_arg,
                      coef_ptr=coef_ptr_arg,
                      input_desc=part_desc)

            opg.free_data_t(X_arg, self.dtype)

        else:

            assert len(X_arys) == 1, "does not support more than one sparse input matrix"
            X_arg = opg.build_data_t([x.data for x in X_arys])
            X_cols = X_arys[0].indices.ptr
            X_row_ids = X_arys[0].indptr.ptr
            X_nnz = sum([x.nnz for x in X_arys])

            # call inheriting class _fit that does all cython pointers and calls
            self._fit(X=[X_arg, X_cols, X_row_ids, X_nnz],
                      y=y_arg,
                      coef_ptr=coef_ptr_arg,
                      input_desc=part_desc)

            for ary in X_arys:
                del ary

        opg.free_rank_size_pair(rank_to_sizes)
        opg.free_part_descriptor(part_desc)
        opg.free_data_t(y_arg, self.dtype)
        return self

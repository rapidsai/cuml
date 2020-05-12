#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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


import ctypes
import cudf
import cuml.common.opg_data_utils_mg as opg
import numpy as np
import rmm

from libc.stdint cimport uintptr_t
from cython.operator cimport dereference as deref

from cuml.common.base import Base
from cuml.common.array import CumlArray
from cuml.common.handle cimport cumlHandle
from cuml.common.opg_data_utils_mg cimport *
from cuml.common import input_to_cuml_array
from cuml.decomposition.utils cimport *


class MGFitMixin(object):

    def fit(self, input_data, n_rows, n_cols, partsToSizes, rank):
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

        X_arys = []
        y_arys = []

        for i in range(len(input_data)):
            if i == 0:
                check_dtype = [np.float32, np.float64]
            else:
                check_dtype = self.dtype

            X_m, _, self.n_cols, _ = \
                input_to_cuml_array(input_data[i][0], check_dtype=check_dtype)
            X_arys.append(X_m)

            if i == 0:
                self.dtype = X_m.dtype

            y_m, *_ = input_to_cuml_array(input_data[i][1],
                                          check_dtype=self.dtype)
            y_arys.append(y_m)

        n_total_parts = len(input_data)

        self._coef_ = CumlArray.zeros(self.n_cols,
                                      dtype=self.dtype)
        cdef uintptr_t coef_ptr = self._coef_.ptr
        coef_ptr_arg = <size_t>coef_ptr

        cdef uintptr_t rank_to_sizes = opg.build_rank_size_pair(input_data,
                                                                rank)
        rank_to_size_arg = <size_t>rank_to_sizes

        cdef uintptr_t data = opg.build_data_t(X_arys)
        X_arg = <size_t>data
        cdef uintptr_t labels = opg.build_data_t(y_arys)
        y_arg = <size_t>labels

        # call inheriting class _fit that does all cython pointers and calls
        self._fit(X=X_arg,
                  y=y_arg,
                  coef_ptr=coef_ptr_arg,
                  rank_to_sizes=rank_to_size_arg,
                  n_rows=n_rows,
                  n_cols=n_cols,
                  n_total_parts=n_total_parts)

        opg.free_rank_size_pair(rank_to_sizes, n_total_parts)

        opg.free_data_t(data, n_total_parts, self.dtype)
        opg.free_data_t(labels, n_total_parts, self.dtype)

        return self

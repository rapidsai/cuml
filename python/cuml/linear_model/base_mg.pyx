#
# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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


import ctypes
import cuml.common.opg_data_utils_mg as opg
import numpy as np
import rmm

from libc.stdint cimport uintptr_t
from cython.operator cimport dereference as deref

import cuml.internals
from cuml.internals.base import Base
from cuml.internals.array import CumlArray
from pylibraft.common.handle cimport handle_t
from cuml.common.opg_data_utils_mg cimport *
from cuml.internals.input_utils import input_to_cuml_array
from cuml.decomposition.utils cimport *


class MGFitMixin(object):

    @cuml.internals.api_base_return_any_skipall
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
        self._set_n_features_in(n_cols)

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

        cdef uintptr_t X_arg = opg.build_data_t(X_arys)
        cdef uintptr_t y_arg = opg.build_data_t(y_arys)

        # call inheriting class _fit that does all cython pointers and calls
        self._fit(X=X_arg,
                  y=y_arg,
                  coef_ptr=coef_ptr_arg,
                  input_desc=part_desc)

        opg.free_rank_size_pair(rank_to_sizes)
        opg.free_part_descriptor(part_desc)
        opg.free_data_t(X_arg, self.dtype)
        opg.free_data_t(y_arg, self.dtype)

        return self

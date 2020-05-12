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


import ctypes
import cudf
import numpy as np

import rmm

from libc.stdlib cimport malloc, free

from libcpp cimport bool
from libc.stdint cimport uintptr_t, uint32_t, uint64_t
from cython.operator cimport dereference as deref

from cuml.common.array import CumlArray
import cuml.common.opg_data_utils_mg as opg

from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
from cuml.decomposition.utils cimport *
from cuml.common import input_to_dev_array, zeros
from cuml.common import input_to_cuml_array
from cuml.common.opg_data_utils_mg cimport *


class BaseDecompositionMG(object):

    def __init__(self, **kwargs):
        super(BaseDecompositionMG, self).__init__(**kwargs)

    def _fit(self, X, total_rows, n_cols, partsToRanks, rank,
             _transform=False):
        """
        Fit function for PCA MG. This not meant to be used as
        part of the public API.
        :param X: array of local dataframes / array partitions
        :param total_rows: total number of rows
        :param n_cols: total number of cols
        :param partsToRanks: array of tuples in the format: [(rank,size)]
        :return: self
        """

        self._set_output_type(X[0])

        X_arys = []
        for i in range(len(X)):
            if i == 0:
                check_dtype = [np.float32, np.float64]
            else:
                check_dtype = self.dtype

            X_m, _, self.n_cols, _ = \
                input_to_cuml_array(X[i], check_dtype=check_dtype)
            X_arys.append(X_m)

            if i == 0:
                self.dtype = X_m.dtype

        cdef uintptr_t data = opg.build_data_t(X_arys)
        X_arg = <size_t>data

        cdef uintptr_t trans_data
        if _transform:
            trans_arys = opg.build_pred_or_trans_arys(X_arys, "F", self.dtype)
            trans_data = opg.build_data_t(trans_arys)
            trans_arg = <size_t> trans_data

        n_total_parts = len(X)

        self._initialize_arrays(self.n_components, total_rows, n_cols)

        cdef uintptr_t rank_to_sizes = opg.build_rank_size_pair(X,
                                                                rank)
        arg_rank_size_pair = <size_t>rank_to_sizes
        decomp_params = self._build_params(total_rows, n_cols)

        if _transform:
            self._call_fit(
                X_arg, trans_arg, rank, arg_rank_size_pair, n_total_parts,
                decomp_params)
        else:
            self._call_fit(X_arg, rank, arg_rank_size_pair,
                           n_total_parts, decomp_params)

        opg.free_rank_size_pair(rank_to_sizes, n_total_parts)
        opg.free_data_t(data, n_total_parts, self.dtype)

        if _transform:
            trans_out = []

            for i in range(len(trans_arys)):
                trans_out.append(trans_arys[i].to_output(
                    output_type=self._get_output_type(X[0])))

            opg.free_data_t(trans_data, n_total_parts, self.dtype)

            return trans_out

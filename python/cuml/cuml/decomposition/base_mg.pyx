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
import cudf
import numpy as np

import rmm

from libc.stdlib cimport malloc, free

from libcpp cimport bool
from libc.stdint cimport uintptr_t, uint32_t, uint64_t
from cython.operator cimport dereference as deref

from cuml.common.array import CumlArray
import cuml.common.opg_data_utils_mg as opg
import cuml.internals
from cuml.common.base import Base
from raft.common.handle cimport handle_t
from cuml.decomposition.utils cimport *
from cuml.decomposition.utils_mg cimport *
from cuml.common import input_to_cuml_array
from cuml.common.opg_data_utils_mg cimport *

from enum import IntEnum


class MGSolver(IntEnum):
    COV_EIG_DQ = <underlying_type_t_solver> mg_solver.COV_EIG_DQ
    COV_EIG_JACOBI = <underlying_type_t_solver> mg_solver.COV_EIG_JACOBI
    QR = <underlying_type_t_solver> mg_solver.QR


class BaseDecompositionMG(object):

    def __init__(self, **kwargs):
        super(BaseDecompositionMG, self).__init__(**kwargs)

    @cuml.internals.api_base_return_any_skipall
    def fit(self, X, total_rows, n_cols, partsToRanks, rank,
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
        self._set_n_features_in(n_cols)

        if self.n_components is None:
            self._n_components = min(total_rows, n_cols)
        else:
            self._n_components = self.n_components

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

        cdef uintptr_t X_arg = opg.build_data_t(X_arys)

        cdef uintptr_t rank_to_sizes = opg.build_rank_size_pair(partsToRanks,
                                                                rank)

        cdef uintptr_t part_desc = opg.build_part_descriptor(total_rows,
                                                             self.n_cols,
                                                             rank_to_sizes,
                                                             rank)

        cdef uintptr_t trans_data
        cdef uintptr_t trans_part_desc
        if _transform:
            trans_arys = opg.build_pred_or_trans_arys(X_arys, "F", self.dtype)
            trans_arg = opg.build_data_t(trans_arys)

            trans_part_desc = opg.build_part_descriptor(total_rows,
                                                        self._n_components,
                                                        rank_to_sizes,
                                                        rank)

        self._initialize_arrays(self._n_components, total_rows, n_cols)
        decomp_params = self._build_params(total_rows, n_cols)

        if _transform:
            self._call_fit(
                X_arg, trans_arg, rank, part_desc, trans_part_desc,
                decomp_params)
        else:
            self._call_fit(X_arg, rank, part_desc, decomp_params)

        opg.free_rank_size_pair(rank_to_sizes)
        opg.free_part_descriptor(part_desc)
        opg.free_data_t(X_arg, self.dtype)

        if _transform:
            trans_out = []

            for i in range(len(trans_arys)):
                trans_out.append(trans_arys[i].to_output(
                    output_type=self._get_output_type(X[0])))

            opg.free_data_t(trans_arg, self.dtype)
            opg.free_part_descriptor(trans_part_desc)

            return trans_out

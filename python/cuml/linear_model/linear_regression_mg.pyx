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
import cuml.utils.opg_data_utils as opg
import numpy as np
import rmm

from libc.stdlib cimport malloc, free

from libcpp cimport bool
from libc.stdint cimport uintptr_t, uint32_t, uint64_t
from cython.operator cimport dereference as deref

from cuml.common.base import Base
from cuml.common.array import CumlArray
from cuml.common.handle cimport cumlHandle
from cuml.decomposition.utils cimport *
from cuml.utils.opg_data_utils cimport *
from cuml.utils import input_to_cuml_array

from cuml.linear_model import LinearRegression


cdef extern from "cumlprims/opg/ols.hpp" namespace "ML::OLS::opg":

    cdef void fit(cumlHandle& handle,
                  RankSizePair **rank_sizes,
                  size_t n_parts,
                  floatData_t **input,
                  size_t n_rows,
                  size_t n_cols,
                  floatData_t **labels,
                  float *coef,
                  float *intercept,
                  bool fit_intercept,
                  bool normalize,
                  int algo,
                  bool verbose) except +

    cdef void fit(cumlHandle& handle,
                  RankSizePair **rank_sizes,
                  size_t n_parts,
                  doubleData_t **input,
                  size_t n_rows,
                  size_t n_cols,
                  doubleData_t **labels,
                  double *coef,
                  double *intercept,
                  bool fit_intercept,
                  bool normalize,
                  int algo,
                  bool verbose) except +


class LinearRegressionMG(LinearRegression):

    def __init__(self, **kwargs):
        super(LinearRegressionMG, self).__init__(**kwargs)

    def fit(self, input_data, n_rows, n_cols, partsToSizes, rank):
        """
        Fit function for MNMG Linear Regression.
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

            X_m, _, self.n_cols, _= \
                input_to_cuml_array(input_data[i][0], check_dtype=check_dtype)
            X_arys.append(X_m)

            if i == 0:
                self.dtype == X_m.dtype

            y_m, *_ = input_to_cuml_array(input_data[i][1],
                                          check_dtype=self.dtype)
            y_arys.append(y_m)

        cdef uintptr_t ranks_sizes = opg.build_rank_size_pair(input_data, rank)

        n_total_parts = len(input_data)

        self._coef_ = CumlArray.zeros(self.n_cols,
                                      dtype=self.dtype)
        cdef uintptr_t coef_ptr = self._coef_.ptr

        cdef float float_intercept
        cdef double double_intercept
        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()
        cdef uintptr_t data
        cdef uintptr_t labels

        if self.dtype == np.float32:
            data = opg.build_data_t(X_arys)
            labels = opg.build_data_t(y_arys)

            fit(handle_[0],
                <RankSizePair**>ranks_sizes,
                <size_t> n_total_parts,
                <floatData_t**>data,
                <size_t>n_rows,
                <size_t>n_cols,
                <floatData_t**>labels,
                <float*>coef_ptr,
                <float*>&float_intercept,
                <bool>self.fit_intercept,
                <bool>self.normalize,
                <int>self.algo,
                False)

            self.intercept_ = float_intercept
        else:
            data = opg.build_data_t(X_arys)
            labels = opg.build_data_t(y_arys)

            fit(handle_[0],
                <RankSizePair**>ranks_sizes,
                <size_t> n_total_parts,
                <doubleData_t**>data,
                <size_t>n_rows,
                <size_t>n_cols,
                <doubleData_t**>labels,
                <double*>coef_ptr,
                <double*>&double_intercept,
                <bool>self.fit_intercept,
                <bool>self.normalize,
                <int>self.algo,
                False)

            self.intercept_ = double_intercept

        self.handle.sync()

        opg.free_data_t(data, n_total_parts, self.dtype)
        opg.free_data_t(labels, n_total_parts, self.dtype)

        opg.free_rank_size_pair(ranks_sizes, n_total_parts)

        return self

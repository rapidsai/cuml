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
import numpy as np

import rmm

from libc.stdlib cimport malloc, free

from libcpp cimport bool
from libc.stdint cimport uintptr_t, uint32_t, uint64_t
from cython.operator cimport dereference as deref

from cuml.common.base import Base
from cuml.common.array import CumlArray
from cuml.common.handle cimport cumlHandle
from cuml.common.opg_data_utils_mg cimport *
from cuml.common import input_to_cuml_array
from cuml.decomposition.utils cimport *

from cuml.linear_model import Ridge
from cuml.linear_model.base_mg import MGFitMixin

cdef extern from "cumlprims/opg/ridge.hpp" namespace "ML::Ridge::opg":

    cdef void fit(cumlHandle& handle,
                  RankSizePair **rank_sizes,
                  size_t n_parts,
                  floatData_t **input,
                  size_t n_rows,
                  size_t n_cols,
                  floatData_t **labels,
                  float *alpha,
                  int n_alpha,
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
                  double *alpha,
                  int n_alpha,
                  double *coef,
                  double *intercept,
                  bool fit_intercept,
                  bool normalize,
                  int algo,
                  bool verbose) except +


class RidgeMG(MGFitMixin, Ridge):

    def __init__(self, **kwargs):
        super(RidgeMG, self).__init__(**kwargs)

    def _fit(self, X, y, coef_ptr, rank_to_sizes, n_rows, n_cols,
             n_total_parts):

        cdef float float_intercept
        cdef double double_intercept
        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()
        cdef float float_alpha
        cdef double double_alpha
        # Only one alpha is supported.
        self.n_alpha = 1

        if self.dtype == np.float32:
            float_alpha = self.alpha

            fit(handle_[0],
                <RankSizePair**><size_t>rank_to_sizes,
                <size_t> n_total_parts,
                <floatData_t**><size_t>X,
                <size_t>n_rows,
                <size_t>n_cols,
                <floatData_t**><size_t>y,
                <float*>&float_alpha,
                <int>self.n_alpha,
                <float*><size_t>coef_ptr,
                <float*>&float_intercept,
                <bool>self.fit_intercept,
                <bool>self.normalize,
                <int>self.algo,
                False)

            self.intercept_ = float_intercept

        else:
            double_alpha = self.alpha

            fit(handle_[0],
                <RankSizePair**><size_t>rank_to_sizes,
                <size_t> n_total_parts,
                <doubleData_t**><size_t>X,
                <size_t>n_rows,
                <size_t>n_cols,
                <doubleData_t**><size_t>y,
                <double*>&double_alpha,
                <int>self.n_alpha,
                <double*><size_t>coef_ptr,
                <double*>&double_intercept,
                <bool>self.fit_intercept,
                <bool>self.normalize,
                <int>self.algo,
                False)

            self.intercept_ = double_intercept

        self.handle.sync()

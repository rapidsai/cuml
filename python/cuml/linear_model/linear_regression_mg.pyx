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
# distutils: language = c++

import ctypes
import cudf
import cuml.common.opg_data_utils_mg as opg
import numpy as np
import rmm

from libc.stdlib cimport malloc, free

from libcpp cimport bool
from libc.stdint cimport uintptr_t, uint32_t, uint64_t
from cython.operator cimport dereference as deref

from cuml.common.base import Base
from cuml.common.array import CumlArray
from cuml.raft.common.handle cimport handle_t
from cuml.common.opg_data_utils_mg cimport *
from cuml.common import input_to_cuml_array
from cuml.decomposition.utils cimport *

from cuml.linear_model import LinearRegression
from cuml.linear_model.base_mg import MGFitMixin


cdef extern from "cuml/linear_model/ols_mg.hpp" namespace "ML::OLS::opg":

    cdef void fit(handle_t& handle,
                  vector[floatData_t *] input_data,
                  PartDescriptor &input_desc,
                  vector[floatData_t *] labels,
                  float *coef,
                  float *intercept,
                  bool fit_intercept,
                  bool normalize,
                  int algo,
                  bool verbose) except +

    cdef void fit(handle_t& handle,
                  vector[doubleData_t *] input_data,
                  PartDescriptor &input_desc,
                  vector[doubleData_t *] labels,
                  double *coef,
                  double *intercept,
                  bool fit_intercept,
                  bool normalize,
                  int algo,
                  bool verbose) except +


class LinearRegressionMG(MGFitMixin, LinearRegression):

    def __init__(self, **kwargs):
        super(LinearRegressionMG, self).__init__(**kwargs)

    def _fit(self, X, y, coef_ptr, input_desc):

        cdef float float_intercept
        cdef double double_intercept
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        if self.dtype == np.float32:

            fit(handle_[0],
                deref(<vector[floatData_t*]*><uintptr_t>X),
                deref(<PartDescriptor*><uintptr_t>input_desc),
                deref(<vector[floatData_t*]*><uintptr_t>y),
                <float*><size_t>coef_ptr,
                <float*>&float_intercept,
                <bool>self.fit_intercept,
                <bool>self.normalize,
                <int>self.algo,
                False)

            self.intercept_ = float_intercept
        else:

            fit(handle_[0],
                deref(<vector[doubleData_t*]*><uintptr_t>X),
                deref(<PartDescriptor*><uintptr_t>input_desc),
                deref(<vector[doubleData_t*]*><uintptr_t>y),
                <double*><size_t>coef_ptr,
                <double*>&double_intercept,
                <bool>self.fit_intercept,
                <bool>self.normalize,
                <int>self.algo,
                False)

            self.intercept_ = double_intercept

        self.handle.sync()

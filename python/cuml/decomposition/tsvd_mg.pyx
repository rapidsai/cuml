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
# distutils: language = c++

import ctypes
import cudf
import numpy as np

import rmm

from libc.stdlib cimport malloc, free

from libcpp cimport bool
from libc.stdint cimport uintptr_t, uint32_t, uint64_t
from cython.operator cimport dereference as deref

import cuml.internals
from cuml.common.base import Base
from cuml.raft.common.handle cimport handle_t
from cuml.decomposition.utils cimport *
import cuml.common.opg_data_utils_mg as opg
from cuml.common.opg_data_utils_mg cimport *

from cuml.decomposition import TruncatedSVD
from cuml.decomposition.base_mg import BaseDecompositionMG

cdef extern from "cuml/decomposition/tsvd_mg.hpp" namespace "ML::TSVD::opg":

    cdef void fit_transform(handle_t& handle,
                            vector[floatData_t *] input_data,
                            PartDescriptor &input_desc,
                            vector[floatData_t *] trans_data,
                            PartDescriptor &trans_desc,
                            float *components,
                            float *explained_var,
                            float *explained_var_ratio,
                            float *singular_vals,
                            paramsTSVD &prms,
                            bool verbose) except +

    cdef void fit_transform(handle_t& handle,
                            vector[doubleData_t *] input_data,
                            PartDescriptor &input_desc,
                            vector[doubleData_t *] trans_data,
                            PartDescriptor &trans_desc,
                            double *components,
                            double *explained_var,
                            double *explained_var_ratio,
                            double *singular_vals,
                            paramsTSVD &prms,
                            bool verbose) except +


class TSVDMG(BaseDecompositionMG, TruncatedSVD):

    def __init__(self, **kwargs):
        super(TSVDMG, self).__init__(**kwargs)

    @cuml.internals.api_base_return_any_skipall
    def _call_fit(self, X, trans, rank, input_desc,
                  trans_desc, arg_params):

        cdef uintptr_t comp_ptr = self.components_.ptr
        cdef uintptr_t explained_var_ptr = self.explained_variance_.ptr
        cdef uintptr_t explained_var_ratio_ptr = \
            self.explained_variance_ratio_.ptr
        cdef uintptr_t singular_vals_ptr = self.singular_values_.ptr
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        cdef paramsTSVD *params = <paramsTSVD*><size_t>arg_params

        if self.dtype == np.float32:

            fit_transform(handle_[0],
                          deref(<vector[floatData_t*]*><uintptr_t>X),
                          deref(<PartDescriptor*><uintptr_t>input_desc),
                          deref(<vector[floatData_t*]*><uintptr_t>trans),
                          deref(<PartDescriptor*><uintptr_t>trans_desc),
                          <float*> comp_ptr,
                          <float*> explained_var_ptr,
                          <float*> explained_var_ratio_ptr,
                          <float*> singular_vals_ptr,
                          deref(params),
                          False)
        else:

            fit_transform(handle_[0],
                          deref(<vector[doubleData_t*]*><uintptr_t>X),
                          deref(<PartDescriptor*><uintptr_t>input_desc),
                          deref(<vector[doubleData_t*]*><uintptr_t>trans),
                          deref(<PartDescriptor*><uintptr_t>trans_desc),
                          <double*> comp_ptr,
                          <double*> explained_var_ptr,
                          <double*> explained_var_ratio_ptr,
                          <double*> singular_vals_ptr,
                          deref(params),
                          False)

        self.handle.sync()

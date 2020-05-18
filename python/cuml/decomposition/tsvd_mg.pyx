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
from cuml.common.handle cimport cumlHandle
from cuml.decomposition.utils cimport *

from cuml.decomposition import TruncatedSVD
from cuml.decomposition.base_mg import BaseDecompositionMG

cdef extern from "cumlprims/opg/matrix/data.hpp" \
                 namespace "MLCommon::Matrix":

    cdef cppclass floatData_t:
        floatData_t(float *ptr, size_t totalSize)
        float *ptr
        size_t totalSize

    cdef cppclass doubleData_t:
        doubleData_t(double *ptr, size_t totalSize)
        double *ptr
        size_t totalSize

cdef extern from "cumlprims/opg/matrix/part_descriptor.hpp" \
                 namespace "MLCommon::Matrix":

    cdef cppclass RankSizePair:
        int rank
        size_t size

cdef extern from "cumlprims/opg/tsvd.hpp" namespace "ML::TSVD::opg":

    cdef void fit_transform(cumlHandle& handle,
                            RankSizePair **rank_sizes,
                            size_t n_parts,
                            floatData_t **input,
                            floatData_t **trans_input,
                            float *components,
                            float *explained_var,
                            float *explained_var_ratio,
                            float *singular_vals,
                            paramsTSVD &prms,
                            bool verbose) except +

    cdef void fit_transform(cumlHandle& handle,
                            RankSizePair **rank_sizes,
                            size_t n_parts,
                            doubleData_t **input,
                            doubleData_t **trans_input,
                            double *components,
                            double *explained_var,
                            double *explained_var_ratio,
                            double *singular_vals,
                            paramsTSVD &prms,
                            bool verbose) except +


class TSVDMG(BaseDecompositionMG, TruncatedSVD):

    def __init__(self, **kwargs):
        super(TSVDMG, self).__init__(**kwargs)

    def _call_fit(self, X, trans, rank, arg_rank_size_pair,
                  n_total_parts, arg_params):

        cdef uintptr_t comp_ptr = self._components_.ptr
        cdef uintptr_t explained_var_ptr = self._explained_variance_.ptr
        cdef uintptr_t explained_var_ratio_ptr = \
            self._explained_variance_ratio_.ptr
        cdef uintptr_t singular_vals_ptr = self._singular_values_.ptr
        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        cdef paramsTSVD *params = <paramsTSVD*><size_t>arg_params

        if self.dtype == np.float32:

            fit_transform(handle_[0],
                          <RankSizePair**><size_t>arg_rank_size_pair,
                          <size_t> n_total_parts,
                          <floatData_t**><size_t> X,
                          <floatData_t**><size_t> trans,
                          <float*> comp_ptr,
                          <float*> explained_var_ptr,
                          <float*> explained_var_ratio_ptr,
                          <float*> singular_vals_ptr,
                          deref(params),
                          False)
        else:

            fit_transform(handle_[0],
                          <RankSizePair**><size_t>arg_rank_size_pair,
                          <size_t> n_total_parts,
                          <doubleData_t**><size_t> X,
                          <doubleData_t**><size_t> trans,
                          <double*> comp_ptr,
                          <double*> explained_var_ptr,
                          <double*> explained_var_ratio_ptr,
                          <double*> singular_vals_ptr,
                          deref(params),
                          False)

        self.handle.sync()

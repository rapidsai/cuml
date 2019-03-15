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

import cudf
import numpy as np

from numba import cuda
from cuml import numba_utils

from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from cuml.hmm.sample_utils import *
from cuml.hmm.hmm_base import _BaseHMM

from libcpp.vector cimport vector


cdef extern from "hmm/hmm_variables.h" namespace "hmm" :
    cdef cppclass HMM[T]:
        pass

cdef extern from "hmm/hmm_py.h" namespace "hmm" nogil:

    cdef void init_f64(HMM[double] &hmm,
                       vector[double*] dmu,
                       vector[double*] dsigma,
                       vector[double*] dPis,
                       vector[double*] dPis_inv,
                       double* dLlhd,
                       double* cur_llhd,
                       int lddx,
                       int lddmu,
                       int lddsigma,
                       int lddsigma_full,
                       int lddPis,
                       int lddLlhd,
                       int nCl,
                       int nDim,
                       int nObs,
                       double reg_covar,
                       int nStates,
                       double* dT,
                       int lddt
                       )


class GMMHMM(_BaseHMM):
    def __init__(self, ):
        super().__init__()

    cdef setup_hmm(self):
        cdef vector[double*] _dmu_ptr_vector
        cdef vector[double*] _dsigma_ptr_vector
        cdef vector[double*] _dPis_ptr_vector
        cdef vector[double*] _dPis_inv_ptr_vector

        for gmm in self.gmms:
            gmm.initialize_parameters()

        for i in range(self.n_mix):
            cdef uintptr_t _dmu_ptr = self.gmms[i].dParams["mus"].device_ctypes_pointer.value
            cdef uintptr_t _dsigma_ptr = self.gmms[i].dParams["sigmas"].device_ctypes_pointer.value
            cdef uintptr_t _dPis_ptr = self.gmms[i].dParams["pis"].device_ctypes_pointer.value
            cdef uintptr_t _dPis_inv_ptr = self.gmms[i].dParams["inv_pis"].device_ctypes_pointer.value

            _dmu_ptr_vector.push_back(_dmu_ptr)
            _dsigma_ptr_vector.push_back(_dsigma_ptr)
            _dPis_ptr_vector.push_back(_dPis_ptr)
            _dPis_inv_ptr_vector.push_back(_dPis_inv_ptr)

        cdef uintptr_t _dB_ptr = self.dB.device_ctypes_pointer.value
        cdef uintptr_t _dT_ptr = self.dT.device_ctypes_pointer.value

        # TODO : Very important remove memory allocatio of dLlhd by each gmm, not used dB used instead
        cdef int lddx = self.gmm[0].ldd["x"]
        cdef int lddmu = self.gmm[0].ldd["mus"]
        cdef int lddsigma = self.gmm[0].ldd["sigmas"]
        cdef int lddsigma_full = self.gmm[0].ldd["sigmas"] * self.nDim
        cdef int lddPis = self.gmm[0].ldd["pis"]
        cdef int lddLlhd =self.gmm[0].ldd["llhd"]
        cdef int lddt = self.lddt

        cdef uintptr_t _cur_llhd_ptr = self.gmm[0].cur_llhd.device_ctypes_pointer.value
        cdef float reg_covar = self.gmm[0].reg_covar

        cdef int nCl = self.nCl
        cdef int nDim = self.nDim
        cdef int nObs = self.nObs
        cdef int nStates = self.n_components

        cdef HMM[float] hmm32
        cdef HMM[double] hmm64

        if self.precision == 'double':
            with nogil:
                init_f64(hmm64,
                         _dmu_ptr_vector,
                         _dsigma_ptr_vector,
                         _dPis_ptr_vector,
                         _dPis_inv_ptr_vector,
                         _dB_ptr,
                         _cur_llhd_ptr,
                         lddx,
                         lddmu,
                         lddsigma,
                         lddsigma_full,
                         lddPis,
                         lddLlhd,
                         nCl,
                         nDim,
                         nObs,
                         reg_covar,
                         nStates,
                         _dT_ptr,
                         lddt)

        return hmm64

    # def predict_proba(self, X, lengths=None):
    #     cdef HMM[float] hmm
    #     hmm = self.setup_hmm()
    #
    #     # TODO: Fix this part
    #     cdef uintptr_t _dX_ptr = self.dParams["x"].device_ctypes_pointer.value
    #
    #     if self.precision == 'single':
    #         with nogil:
    #             forward_f32(hmm, <float*> _dX_ptr)

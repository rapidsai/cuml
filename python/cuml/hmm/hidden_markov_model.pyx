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
from cuml.hmm.hmm_base import HMMBase

from libcpp.vector cimport vector


cdef extern from "hmm/hmm_variables.h" :
    cdef cppclass HMM[T]:
        pass

cdef extern from "hmm/hmm_py.h" nogil:

    cdef void init_f32(HMM[float]&,
                       )
    cdef void setup_f32(HMM[float]&)

    cdef void forward_f32(HMM[float]&,
                          float*,
                          int*,
                          int)
    cdef void backward_f32(HMM[float]&,
                          float*,
                          int*,
                          int)

class GMMHMM(HMMBase):
    def __init__(self, ):
        super().__init__()

    cdef setup_hmm(self):
        cdef vector[double*] *_dmu_ptr_vector = new vector[double*]()
        cdef int i

        for gmm in self.gmms:
            gmm.initialize_parameters()

        for i in range(self.n_mix):
            cdef uintptr_t _dmu_ptr = self.gmms[i].dParams["mus"].device_ctypes_pointer.value
            cdef uintptr_t _dsigma_ptr = self.gmms[i].dParams["sigmas"].device_ctypes_pointer.value
            cdef uintptr_t _dPis_ptr = self.gmms[i].dParams["pis"].device_ctypes_pointer.value
            cdef uintptr_t _dPis_inv_ptr = self.gmms[i].dParams["inv_pis"].device_ctypes_pointer.value

            _dmu_ptr_vector.push_back(_dmu_ptr)
            _dmu_ptr_vector.push_back(_dsigma_ptr)
            _dPis_ptr.push_back(_dPis_ptr)
            _dPis_inv_ptr.push_back(_dPis_inv_ptr)

        cdef HMM[float] hmm32

        if self.precision == 'single':
            with nogil:
                init_f32()

        return hmm32

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

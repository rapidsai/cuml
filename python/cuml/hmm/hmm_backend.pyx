# Copyright (c) 2018, NVIDIA CORPORATION.
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

from libc.stdint cimport uintptr_t
from libc.stdlib cimport  malloc
from libcpp.vector cimport vector
from libcpp cimport bool

from cuml.hmm.hmm_extern cimport *
from cuml.gmm.gaussian_mixture_backend cimport *
from cuml.hmm._multinomial cimport *

from numba import cuda
import numpy as np
from cuml.gmm.utils.utils import to_gb, to_mb

RUP_SIZE = 32


# cdef setup_gmmhmm(self, floatGMMHMM& hmm32, doubleGMMHMM& hmm64):
#     cdef vector[GMM[float]] gmms32
#     cdef vector[GMM[double]] gmms64
#
#     cdef GMM[float] *ar_gmms32 = <GMM[float] *>malloc(self.n_components * sizeof(GMM[float]))
#     cdef GMM[double] *ar_gmms64 = <GMM[double] *>malloc(self.n_components * sizeof(GMM[double]))
#
#     cdef uintptr_t _dB_ptr = self.dB.device_ctypes_pointer.value
#     cdef uintptr_t _dT_ptr = self.dT.device_ctypes_pointer.value
#     cdef uintptr_t _dGamma_ptr = self.dGamma.device_ctypes_pointer.value
#
#     cdef int nStates = self.n_components
#     cdef int lddt = self.lddt
#     cdef int lddb = self.lddb
#     cdef int lddgamma = self.lddgamma
#
#     if self.precision == 'double':
#         for i in range(self.n_components):
#             _setup_gmm(self.gmms[i], ar_gmms32[i], ar_gmms64[i])
#             gmms64.push_back(ar_gmms64[i])
#
#         with nogil:
#             init_f64(hmm64,
#                      gmms64,
#                      <int>nStates,
#                      <double*>_dT_ptr,
#                      <int>lddt,
#                      <double*>_dB_ptr,
#                      <int>lddb,
#                      <double*>_dGamma_ptr,
#                      <int>lddgamma)

cdef setup_multinomialhmm(self, floatMultinomialHMM& hmm32, doubleMultinomialHMM& hmm64, bool do_handle):
    cdef vector[Multinomial[float]] multinomials32
    cdef vector[Multinomial[double]] multinomials64

    cdef Multinomial[float] *ar_multinomials32 = <Multinomial[float] *>malloc(self.n_components * sizeof(Multinomial[float]))
    cdef Multinomial[double] *ar_multinomials64 = <Multinomial[double] *>malloc(self.n_components * sizeof(Multinomial[double]))

    cdef uintptr_t _dB_ptr = self.dB.device_ctypes_pointer.value
    cdef uintptr_t _dT_ptr = self.dT.device_ctypes_pointer.value
    cdef uintptr_t _dGamma_ptr = self.dGamma.device_ctypes_pointer.value
    cdef uintptr_t _dStartProb_ptr = self.dstartProb.device_ctypes_pointer.value
    cdef uintptr_t _dLlhd_ptr = self.dLlhd.device_ctypes_pointer.value

    cdef uintptr_t _logllhd_ptr = self.dlogllhd.device_ctypes_pointer.value
    cdef uintptr_t _ws_ptr

    cdef size_t size

    cdef int nStates = self.n_components
    cdef int lddt = self.lddt
    cdef int lddb = self.lddb
    cdef int lddgamma = self.lddgamma
    cdef int lddsp = self.lddsp

    cdef int nObs = self.nObs
    cdef int nSeq = self.nSeq

    if self.precision == 'single':
        for i in range(self.n_components):
            _setup_multinomial(self.dists[i], ar_multinomials32[i], ar_multinomials64[i])
            multinomials32.push_back(ar_multinomials32[i])

        with nogil:
            init_mhmm_f32(hmm32,
                          multinomials32,
                          <int>nStates,
                          <float*>_dStartProb_ptr,
                          <int>lddsp,
                          <float*>_dT_ptr,
                          <int>lddt,
                          <float*>_dB_ptr,
                          <int>lddb,
                          <float*>_dGamma_ptr,
                          <int>lddgamma,
                          <float*>_logllhd_ptr,
                          <int> nObs,
                          <int> nSeq,
                          <float*> _dLlhd_ptr)

    if self.precision == 'double':
        for i in range(self.n_components):
            _setup_multinomial(self.dists[i], ar_multinomials32[i], ar_multinomials64[i])
            multinomials64.push_back(ar_multinomials64[i])

        with nogil:
            init_mhmm_f64(hmm64,
                          multinomials64,
                          <int>nStates,
                          <double*>_dStartProb_ptr,
                          <int>lddsp,
                          <double*>_dT_ptr,
                          <int>lddt,
                          <double*>_dB_ptr,
                          <int>lddb,
                          <double*>_dGamma_ptr,
                          <int>lddgamma,
                          <double*>_logllhd_ptr,
                          <int> nObs,
                          <int> nSeq,
                          <double*> _dLlhd_ptr)

    if do_handle :
        _ws_ptr = self.workspace.device_ctypes_pointer.value

        if self.precision == 'single':
            with nogil :
                size = get_workspace_size_mhmm_f32(hmm32)
                create_handle_mhmm_f32(hmm32, <void*> _ws_ptr)

        if self.precision == 'double':
            with nogil :
                size = get_workspace_size_mhmm_f64(hmm64)
                create_handle_mhmm_f64(hmm64, <void*> _ws_ptr)

class _BaseHMMBackend:
    def allocate_ws(self):
        self._workspaceSize = -1
        cdef size_t workspace_size = 0

        cdef floatMultinomialHMM multinomialhmm32
        cdef doubleMultinomialHMM multinomialhmm64

        if self.hmm_type is 'multinomial':
            setup_multinomialhmm(self, multinomialhmm32, multinomialhmm64, False)

        cuda_context = cuda.current_context()
        available_mem = cuda_context.get_memory_info().free
        print('available mem before allocation', to_mb(available_mem), "Mb")

        if self.precision is "single" and self.hmm_type is 'multinomial':
            with nogil :
                workspace_size = get_workspace_size_mhmm_f32(multinomialhmm32)

        if self.precision is "double" and self.hmm_type is 'multinomial':
            with nogil :
                workspace_size = get_workspace_size_mhmm_f64(multinomialhmm64)

        self._workspace_size = workspace_size
        self.workspace = cuda.to_device(np.zeros(self._workspace_size, dtype=np.int8))

        print("----------------\n")
        print('Workspace size', to_mb(self._workspace_size), "Mb")
        print("----------------\n")

        cuda_context = cuda.current_context()
        available_mem = cuda_context.get_memory_info().free
        print('available mem after allocation', to_mb(available_mem), "Mb")

    def _forward_backward(self, X, lengths, do_forward, do_backward, do_gamma):
        cdef floatGMMHMM gmmhmm32
        cdef doubleGMMHMM gmmhmm64

        cdef floatMultinomialHMM multinomialhmm32
        cdef doubleMultinomialHMM multinomialhmm64

        if self.hmm_type is 'multinomial':
            setup_multinomialhmm(self, multinomialhmm32, multinomialhmm64, True)

        cdef uintptr_t _dX_ptr = self.dX.device_ctypes_pointer.value
        cdef uintptr_t _dlengths_ptr = self.dlengths.device_ctypes_pointer.value

        cdef int nObs = self.nObs
        cdef int nSeq = self.nSeq

        cdef bool doForward = do_forward
        cdef bool doBackward = do_backward
        cdef bool doGamma = do_gamma

        if self.precision is "single" and self.hmm_type is 'multinomial':
            forward_backward_mhmm_f32(multinomialhmm32,
                                      <unsigned short int*> _dX_ptr,
                                      <unsigned short int*> _dlengths_ptr,
                                      <int> nSeq,
                                      <bool> doForward,
                                      <bool> doBackward,
                                      <bool> doGamma)

        if self.precision is "double" and self.hmm_type is 'multinomial':
            forward_backward_mhmm_f64(multinomialhmm64,
                                      <unsigned short int*> _dX_ptr,
                                      <unsigned short int*> _dlengths_ptr,
                                      <int> nSeq,
                                      <bool> doForward,
                                      <bool> doBackward,
                                      <bool> doGamma)

    def _viterbi(self, X, lengths):
        cdef floatGMMHMM gmmhmm32
        cdef doubleGMMHMM gmmhmm64

        cdef floatMultinomialHMM multinomialhmm32
        cdef doubleMultinomialHMM multinomialhmm64

        if self.hmm_type is 'multinomial':
            setup_multinomialhmm(self, multinomialhmm32, multinomialhmm64 ,True)

        cdef uintptr_t _dX_ptr = self.dX.device_ctypes_pointer.value
        cdef uintptr_t _dlengths_ptr = self.dlengths.device_ctypes_pointer.value
        cdef uintptr_t _dVStates_ptr = self.dVStates.device_ctypes_pointer.value

        cdef int nSeq = self.nSeq

        if self.precision is "single" and self.hmm_type is 'multinomial':
            viterbi_mhmm_f32(multinomialhmm32,
                             <unsigned short int*> _dVStates_ptr,
                             <unsigned short int*> _dX_ptr,
                             <unsigned short int*> _dlengths_ptr,
                             <int> nSeq)
        if self.precision is "double" and self.hmm_type is 'multinomial':
            viterbi_mhmm_f64(multinomialhmm64,
                             <unsigned short int*> _dVStates_ptr,
                             <unsigned short int*> _dX_ptr,
                             <unsigned short int*> _dlengths_ptr,
                             <int> nSeq)

    def _m_step(self, X, lengths):
        cdef floatGMMHMM gmmhmm32
        cdef doubleGMMHMM gmmhmm64

        cdef floatMultinomialHMM multinomialhmm32
        cdef doubleMultinomialHMM multinomialhmm64

        if self.hmm_type is 'multinomial':
            setup_multinomialhmm(self, multinomialhmm32, multinomialhmm64, True)

        cdef uintptr_t _dX_ptr = self.dX.device_ctypes_pointer.value
        cdef uintptr_t _dlengths_ptr = self.dlengths.device_ctypes_pointer.value

        cdef int nSeq = self.nSeq

        if self.precision is "single" and self.hmm_type is 'multinomial':
            m_step_mhmm_f32(multinomialhmm32,
                            <unsigned short int*> _dX_ptr,
                            <unsigned short int*> _dlengths_ptr,
                            <int> nSeq)
        if self.precision is "double" and self.hmm_type is 'multinomial':
            m_step_mhmm_f64(multinomialhmm64,
                            <unsigned short int*> _dX_ptr,
                            <unsigned short int*> _dlengths_ptr,
                            <int> nSeq)
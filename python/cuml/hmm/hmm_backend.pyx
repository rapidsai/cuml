from libc.stdint cimport uintptr_t
from libc.stdlib cimport  malloc
from libcpp.vector cimport vector
from libcpp cimport bool

from cuml.hmm.hmm_extern cimport *
from cuml.gmm.gaussian_mixture cimport *
from cuml.hmm._multinomial cimport *


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

cdef setup_multinomialhmm(self, floatMultinomialHMM& hmm32, doubleMultinomialHMM& hmm64):
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

    cdef int nStates = self.n_components
    cdef int lddt = self.lddt
    cdef int lddb = self.lddb
    cdef int lddgamma = self.lddgamma
    cdef int lddsp = self.lddsp

    cdef int nObs = self.nObs
    cdef int nSeq = self.nSeq
    # cdef int nFeatures = self.nFeatures

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
                          <double*>_logllhd_ptr)
            setup_mhmm_f64(hmm64,
                           <int> nObs,
                           <int> nSeq,
                           <double*> _dLlhd_ptr)

class _BaseHMMBackend:
    def __init__(self):
        pass

    def _forward_backward(self, X, lengths, do_forward, do_backward, do_gamma):
        self._setup(X, lengths)

        cdef floatGMMHMM gmmhmm32
        cdef doubleGMMHMM gmmhmm64

        cdef floatMultinomialHMM multinomialhmm32
        cdef doubleMultinomialHMM multinomialhmm64

        # if self.hmm_type is "gmm" :
        # setup_gmmhmm(self, gmmhmm32, gmmhmm64)
        if self.hmm_type is 'multinomial':
            setup_multinomialhmm(self, multinomialhmm32, multinomialhmm64)

        cdef uintptr_t _dX_ptr = self.dX.device_ctypes_pointer.value
        cdef uintptr_t _dlengths_ptr = self.dlengths.device_ctypes_pointer.value

        cdef int nObs = self.nObs
        cdef int nSeq = self.nSeq

        cdef bool doForward = do_forward
        cdef bool doBackward = do_backward
        cdef bool doGamma = do_gamma

        # if self.dtype is "double" and self.hmm_type is 'gmm':
        #     forward_backward_f64(gmmhmm64,
        #                          <double*> _dX_ptr,
        #                          <int*> _dlengths_ptr,
        #                          <int> nSeq,
        #                          <bool> doForward,
        #                          <bool> doBackward)

        if self.precision is "double" and self.hmm_type is 'multinomial':
            forward_backward_mhmm_f64(multinomialhmm64,
                                      <unsigned short int*> _dX_ptr,
                                      <unsigned short int*> _dlengths_ptr,
                                      <int> nSeq,
                                      <bool> doForward,
                                      <bool> doBackward,
                                      <bool> doGamma)

    def _viterbi(self, X, lengths):
        self._setup(X, lengths)

        cdef floatGMMHMM gmmhmm32
        cdef doubleGMMHMM gmmhmm64

        cdef floatMultinomialHMM multinomialhmm32
        cdef doubleMultinomialHMM multinomialhmm64

        if self.hmm_type is 'multinomial':
            setup_multinomialhmm(self, multinomialhmm32, multinomialhmm64)

        cdef uintptr_t _dX_ptr = self.dX.device_ctypes_pointer.value
        cdef uintptr_t _dlengths_ptr = self.dlengths.device_ctypes_pointer.value
        cdef uintptr_t _dVStates_ptr = self.dVStates.device_ctypes_pointer.value

        cdef int nSeq = self.nSeq

        if self.precision is "double" and self.hmm_type is 'multinomial':
            viterbi_mhmm_f64(multinomialhmm64,
                             <unsigned short int*> _dVStates_ptr,
                                      <unsigned short int*> _dX_ptr,
                                      <unsigned short int*> _dlengths_ptr,
                                      <int> nSeq)

    def _m_step(self, X, lengths):
        self._setup(X, lengths)

        cdef floatGMMHMM gmmhmm32
        cdef doubleGMMHMM gmmhmm64

        cdef floatMultinomialHMM multinomialhmm32
        cdef doubleMultinomialHMM multinomialhmm64

        if self.hmm_type is 'multinomial':
            setup_multinomialhmm(self, multinomialhmm32, multinomialhmm64)

        cdef uintptr_t _dX_ptr = self.dX.device_ctypes_pointer.value
        cdef uintptr_t _dlengths_ptr = self.dlengths.device_ctypes_pointer.value

        cdef int nSeq = self.nSeq

        if self.precision is "double" and self.hmm_type is 'multinomial':
            m_step_mhmm_f64(multinomialhmm64,
                                      <unsigned short int*> _dX_ptr,
                                      <unsigned short int*> _dlengths_ptr,
                                      <int> nSeq)
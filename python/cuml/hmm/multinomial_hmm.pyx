from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
from libcpp.vector cimport vector
from libcpp cimport bool


from cuml.hmm.hidden_markov_model cimport HMM, init_f64, forward_backward_f64


cdef setup_hmm(self, HMM[float]& hmm32, HMM[double]& hmm64):
    cdef vector[GMM[float]] gmms32
    cdef vector[GMM[double]] gmms64

    cdef GMM[float] *ar_gmms32 = <GMM[float] *>malloc(self.n_components * sizeof(GMM[float]))
    cdef GMM[double] *ar_gmms64 = <GMM[double] *>malloc(self.n_components * sizeof(GMM[double]))

    cdef uintptr_t _dB_ptr = self.dB.device_ctypes_pointer.value
    cdef uintptr_t _dT_ptr = self.dT.device_ctypes_pointer.value
    cdef uintptr_t _dGamma_ptr = self.dGamma.device_ctypes_pointer.value

    cdef int nStates = self.n_components
    cdef int lddt = self.lddt
    cdef int lddb = self.lddb
    cdef int lddgamma = self.lddgamma

    if self.precision == 'double':
        for i in range(self.n_components):
            _setup_gmm(self.gmms[i], ar_gmms32[i], ar_gmms64[i])
            gmms64.push_back(ar_gmms64[i])

        with nogil:
            init_f64(hmm64,
                     gmms64,
                     <int>nStates,
                     <double*>_dT_ptr,
                     <int>lddt,
                     <double*>_dB_ptr,
                     <int>lddb,
                     <double*>_dGamma_ptr,
                     <int>lddgamma)

class HiddenMarkovModel(_BaseHMM, _DevHMM):
    def __init__(self,
                 n_components,
                 n_mix,
                 precision="double",
                 covariance_type="full",
                 random_state=None):
        pass

        super().__init__(n_components=n_components,
                         n_mix=n_mix,
                         precision=precision,
                         random_state=random_state)


    def _fit(self, X, lengths=None):
        pass



    def _forward_backward(self, X, lengths, do_forward, do_backward):
        self._set_dims(X, lengths)
        self._initialize()
        self._setup(X, lengths)

        cdef HMM[float] hmm32
        cdef HMM[double] hmm64
        setup_hmm(self, hmm32, hmm64)

        cdef uintptr_t _dX_ptr = self.dX.device_ctypes_pointer.value
        cdef uintptr_t _dlengths_ptr = self.dlengths.device_ctypes_pointer.value
        cdef int nObs = self.nObs
        cdef int nSeq = self.nSeq

        cdef bool doForward = do_forward
        cdef bool doBackward = do_backward

        for gmm in self.gmms :
            gmm.init_step()

        if self.dtype is "double" :
            forward_backward_f64(hmm64,
                                   <double*> _dX_ptr,
                                   <int*> _dlengths_ptr,
                                   <int> nSeq,
                                   <bool> doForward,
                                   <bool> doBackward)

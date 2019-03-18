import cudf
import numpy as np

from numba import cuda
from cuml import numba_utils

from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free
from libcpp.vector cimport vector

from cuml.hmm.sample_utils import *
from cuml.hmm.gmm_base import _BaseGMM
from cuml.hmm.hmm_base import _BaseHMM
from cuml.hmm.devtools import _DevHMM



cdef extern from "gmm/gmm_variables.h" namespace "gmm":
    cdef cppclass GMM[T]:
        pass

cdef extern from "gmm/gmm_py.h" namespace "gmm" nogil:

    cdef void init_f32(GMM[float]&,
                       float*,
                       float*,
                       float*,
                       float*,
                       float*,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       float*,
                       float,
                       int,
                       int,
                       int)

    cdef void setup_f32(GMM[float]&)
    cdef void compute_lbow_f32(GMM[float]&)
    cdef void update_llhd_f32(float*, GMM[float]&)
    cdef void update_rhos_f32(GMM[float]&, float*)
    cdef void update_mus_f32(float*, GMM[float]&)
    cdef void update_sigmas_f32(float*, GMM[float]&)
    cdef void update_pis_f32(GMM[float]&)

    cdef void init_f64(GMM[double]&,
                       double*,
                       double*,
                       double*,
                       double*,
                       double*,
                       int,
                       int,
                       int,
                       int,
                       int,
                       int,
                       double*,
                       double,
                       int,
                       int,
                       int)

    cdef void setup_f64(GMM[double]&)
    cdef void compute_lbow_f64(GMM[double]&)
    cdef void update_llhd_f64(double*, GMM[double]&)
    cdef void update_rhos_f64(GMM[double]&, double*)
    cdef void update_mus_f64(double*, GMM[double]&)
    cdef void update_sigmas_f64(double*, GMM[double]&)
    cdef void update_pis_f64(GMM[double]&)

cdef extern from "hmm/hmm_variables.h" namespace "hmm":
    cdef cppclass HMM[T]:
        pass

cdef extern from "hmm/hmm_py.h" namespace "hmm" nogil:
    cdef void init_f64(HMM[double] &hmm,
                       vector[GMM[double]] &gmms,
                       int nStates,
                       double* dT,
                       int lddt,
                       double* dB,
                       int lddb,
                       double* dGamma,
                       int lddgamma)

    cdef void forward_backward_f64(HMM[double] &hmm,
                                   double* dX,
                                   int* dlenghts,
                                   int nSeq,
                                   bool doForward,
                                   bool doBackward)


RUP_SIZE = 32

cdef _setup_gmm(self, GMM[float]& gmm32, GMM[double]& gmm64, toAllocate=True):
    cdef uintptr_t _dmu_ptr = self.dParams["mus"].device_ctypes_pointer.value
    cdef uintptr_t _dsigma_ptr = self.dParams["sigmas"].device_ctypes_pointer.value
    cdef uintptr_t _dPis_ptr = self.dParams["pis"].device_ctypes_pointer.value
    cdef uintptr_t _dPis_inv_ptr = self.dParams["inv_pis"].device_ctypes_pointer.value
    cdef uintptr_t _dLlhd_ptr = self.dLlhd.device_ctypes_pointer.value

    cdef int lddx = self.lddx
    cdef int lddmu = self.ldd["mus"]
    cdef int lddsigma = self.ldd["sigmas"]
    cdef int lddsigma_full = self.ldd["sigmas"] * self.nDim
    cdef int lddPis = self.ldd["pis"]
    cdef int lddLlhd =self.lddllhd

    cdef uintptr_t _cur_llhd_ptr = self.cur_llhd.device_ctypes_pointer.value
    cdef float reg_covar = self.reg_covar

    cdef int nCl = self.nCl
    cdef int nDim = self.nDim
    cdef int nObs = self.nObs

    if self.precision == 'single':
        with nogil:
            init_f32(gmm32,
                     <float*> _dmu_ptr,
                     <float*> _dsigma_ptr,
                     <float*> _dPis_ptr,
                     <float*> _dPis_inv_ptr,
                     <float*> _dLlhd_ptr,
                     <int> lddx,
                     <int> lddmu,
                     <int> lddsigma,
                     <int> lddsigma_full,
                     <int> lddPis,
                     <int> lddLlhd,
                     <float*> _cur_llhd_ptr,
                     <float> reg_covar,
                     <int> nCl,
                     <int> nDim,
                     <int> nObs)

    if self.precision == 'double':
        with nogil:
            init_f64(gmm64,
                     <double*> _dmu_ptr,
                     <double*> _dsigma_ptr,
                     <double*> _dPis_ptr,
                     <double*> _dPis_inv_ptr,
                     <double*> _dLlhd_ptr,
                     <int> lddx,
                     <int> lddmu,
                     <int> lddsigma,
                     <int> lddsigma_full,
                     <int> lddPis,
                     <int> lddLlhd,
                     <double*> _cur_llhd_ptr,
                     <double> reg_covar,
                     <int> nCl,
                     <int> nDim,
                     <int> nObs)
    if toAllocate :
        if self.precision == 'single':
            with nogil:
                setup_f32(gmm32)
        if self.precision == 'double':
            with nogil:
                setup_f64(gmm64)


class GaussianMixture(_BaseGMM):
    def __init__(self, n_components, tol=1e-03,
                 reg_covar=1e-06, max_iter=100, init_params="random",
                 warm_start=False, precision='single', random_state=None):
        super().__init__(n_components=n_components,
                         tol=tol,
                         reg_covar=reg_covar,
                         max_iter=max_iter,
                         init_params=init_params,
                         warm_start=warm_start,
                         precision=precision,
                         random_state=random_state)


    def step(self):
        cdef uintptr_t _dX_ptr = self.dX.device_ctypes_pointer.value

        cdef GMM[float] gmm32
        cdef GMM[double] gmm64
        _setup_gmm(self, gmm32, gmm64)

        if self.precision == 'single':
            with nogil:
                update_llhd_f32(<float*>_dX_ptr, gmm32)
                update_rhos_f32(gmm32, <float*> _dX_ptr)

                update_pis_f32(gmm32)
                update_mus_f32(<float*>_dX_ptr, gmm32)
                update_sigmas_f32(<float*>_dX_ptr, gmm32)

                compute_lbow_f32(gmm32)

        if self.precision == 'double':
            with nogil:
                update_llhd_f64(<double*>_dX_ptr, gmm64)
                update_rhos_f64(gmm64, <double*> _dX_ptr)

                update_pis_f64(gmm64)
                update_mus_f64(<double*>_dX_ptr, gmm64)
                update_sigmas_f64(<double*>_dX_ptr, gmm64)

                compute_lbow_f64(gmm64)


    def init_step(self):
        cdef uintptr_t _dX_ptr = self.dX.device_ctypes_pointer.value

        cdef GMM[float] gmm32
        cdef GMM[double] gmm64
        _setup_gmm(self, gmm32, gmm64)

        if self.precision == 'single':
            with nogil:
                update_pis_f32(gmm32)
                update_mus_f32(<float*>_dX_ptr, gmm32)
                update_sigmas_f32(<float*>_dX_ptr, gmm32)

        if self.precision == 'double':
            with nogil:
                update_pis_f64(gmm64)
                update_mus_f64(<double*>_dX_ptr, gmm64)
                update_sigmas_f64(<double*>_dX_ptr, gmm64)

    def _set_dims(self, X=None, nCl=None, nDim=None, nObs=None):
        if X is None :
            self.nCl = nCl
            self.nDim = nDim
            self.nObs = nObs
        else :
            self.nCl = self.n_components
            self.nDim = X.shape[1]
            self.nObs = X.shape[0]

    def _initialize(self):
        self.ldd = {"mus" : roundup(self.nDim, RUP_SIZE),
                    "sigmas" : roundup(self.nDim, RUP_SIZE),
                    "pis" : roundup(self.nCl, RUP_SIZE),
                    "inv_pis" : roundup(self.nCl, RUP_SIZE)}

        params = dict({"mus" : np.zeros((self.ldd["mus"], self.nCl)),
                       "sigmas" : np.zeros((self.ldd["sigmas"] * self.nDim, self.nCl)),
                       "pis" : np.zeros((self.ldd["pis"], 1)),
                       "inv_pis" : np.zeros((self.ldd["inv_pis"], 1))})

        params = align_parameters(params, self.ldd)
        params = flatten_parameters(params)
        params = cast_parameters(params, self.dtype)
        self.dParams = dict(
            (key, cuda.to_device(params[key])) for key in self.ldd.keys())
        self.cur_llhd = cuda.to_device(np.zeros(1, dtype=self.dtype))

    def _setup(self, X):
        self.dX = X.T
        self.dLlhd = sample_matrix(self.nObs, self.nCl, self.random_state, isRowNorm=True)
        self.dLlhd = self.dLlhd.T

        self.lddx = roundup(self.nDim, RUP_SIZE)
        self.lddllhd = roundup(self.nCl, RUP_SIZE)

        # Align flatten, cast and copy to device
        self.dX = process_parameter(self.dX, self.lddx, self.dtype)
        self.dLlhd = process_parameter(self.dLlhd, self.lddllhd, self.dtype)

    def fit(self, X):
        if self.warm_start :
            try:
                getattr(self, "nCl")
            except AttributeError:
                print("Please run the model a first time")
        else :
            self._set_dims(X)
            self._initialize()
        self._setup(X)
        self.init_step()

        prev_lbow = - np.inf

        for it in range(1, self.max_iter + 1) :
            self.step()

            diff = self.lower_bound_ - prev_lbow
            if  diff < self.tol :
                break
            prev_lbow = self.lower_bound_




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

    def _initialize(self):
        # Align flatten, cast and copy to device
        self.dT = sample_matrix(self.n_components, self.n_mix, random_state=self.random_state, isRowNorm=True)
        self.lddt = roundup(self.n_components, RUP_SIZE)
        self.dT = process_parameter(self.dT, self.lddt, self.dtype)

        self.gmms = [GaussianMixture(n_components=self.nCl,
                                     precision=self.precision) for _ in range(self.n_components)]
        for gmm in self.gmms:
            gmm._set_dims(nCl=self.nCl, nDim=self.nDim, nObs=self.nObs)
            gmm._initialize()

    def _set_dims(self, X, lengths):
        self.nObs = X.shape[0]
        self.nDim = X.shape[1]
        self.nCl = self.n_mix
        self.nStates = self.n_components

        if lengths is None :
            self.n_seq = 1
        else :
            self.n_seq = lengths.shape[0]

    def _setup(self, X, lengths):
        self.dB = sample_matrix(self.n_components,
                                self.nObs * self.nCl,
                                random_state=self.random_state,
                                isColNorm=True)
        self.lddb = roundup(self.nCl, RUP_SIZE)
        self.dB = process_parameter(self.dB, self.lddb, self.dtype)

        self.dGamma = np.zeros((self.nStates, self.nObs), dtype=self.dtype)
        self.lddgamma = roundup(self.nStates, RUP_SIZE)
        self.dGamma = process_parameter(self.dGamma, self.lddgamma, self.dtype)

        for gmm in self.gmms :
            gmm._setup(X)

        self.dX = X.T
        self.lddx = roundup(self.nDim, RUP_SIZE)
        # Align flatten, cast and copy to device
        self.dX = process_parameter(self.dX, self.lddx, self.dtype)

        # Process lengths
        if lengths is None :
            lengths = np.array([self.nObs])
        # Check leading dimension
        lengths = lengths.astype(int)
        self.dlengths = cuda.to_device(lengths)

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

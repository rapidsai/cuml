import cudf
import numpy as np

from numba import cuda
from cuml import numba_utils

from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from cuml.hmm.sample_utils import *
from cuml.hmm.gmm_base import _BaseGMM

cdef extern from "hmm/hmm_variables.h" namespace "gmm":
    cdef cppclass GMM[T]:
        pass

cdef extern from "hmm/gmm_py.h" namespace "gmm" nogil:

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

    def _initialize(self, X):
            self.nCl = self.n_components
            self.nDim = X.shape[1]
            self.nObs = X.shape[0]

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
            self._initialize(X)
        self._setup(X)
        self.init_step()

        prev_lbow = - np.inf

        for it in range(1, self.max_iter + 1) :
            self.step()

            diff = self.lower_bound_ - prev_lbow
            if  diff < self.tol :
                break
            prev_lbow = self.lower_bound_

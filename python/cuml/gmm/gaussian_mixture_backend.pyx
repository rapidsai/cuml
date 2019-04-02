from libc.stdint cimport uintptr_t

from cuml.gmm.gaussian_mixture_extern cimport *


RUP_SIZE = 32

cdef _setup_gmm(self, GMM[float]& gmm32, GMM[double]& gmm64, toAllocate=True):
    cdef uintptr_t _dmu_ptr = self.dmus.device_ctypes_pointer.value
    cdef uintptr_t _dsigma_ptr = self.dsigmas.device_ctypes_pointer.value
    cdef uintptr_t _dPis_ptr = self.dpis.device_ctypes_pointer.value
    cdef uintptr_t _dPis_inv_ptr = self.dinv_pis.device_ctypes_pointer.value
    cdef uintptr_t _dLlhd_ptr = self.dLlhd.device_ctypes_pointer.value

    cdef int lddx = self.lddx
    cdef int lddmu = self.lddmus
    cdef int lddsigma = self.lddsigmas
    cdef int lddsigma_full = self.lddsigmas * self.nDim
    cdef int lddPis = self.lddpis
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


class _GaussianMixtureBackend :
    def __init__(self) :
        pass

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



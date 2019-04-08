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

from libc.stdint cimport uintptr_t
from libcpp cimport bool

from cuml.gmm.gaussian_mixture_extern cimport *
from cuml.gmm.utils.utils import to_gb, to_mb

from numba import cuda
import numpy as np

RUP_SIZE = 32

cdef _setup_gmm(self, GMM[float]& gmm32, GMM[double]& gmm64, bool do_handle):
    cdef uintptr_t _dmu_ptr = self.dmus.device_ctypes_pointer.value
    cdef uintptr_t _dsigma_ptr = self.dsigmas.device_ctypes_pointer.value
    cdef uintptr_t _dPis_ptr = self.dpis.device_ctypes_pointer.value
    cdef uintptr_t _dPis_inv_ptr = self.dinv_pis.device_ctypes_pointer.value
    cdef uintptr_t _dLlhd_ptr = self.dLlhd.device_ctypes_pointer.value

    cdef uintptr_t _ws_ptr

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

    if do_handle :
        _ws_ptr = self.workspace.device_ctypes_pointer.value
        if self.precision == 'single':
            with nogil:
                _ = get_workspace_size_f32(gmm32)
                create_gmm_handle_f32(gmm32, <void*> _ws_ptr)
        if self.precision == 'double':
            with nogil:
                _ = get_workspace_size_f64(gmm64)
                create_gmm_handle_f64(gmm64, <void*> _ws_ptr)

    # if self.precision == 'single':
    #     with nogil:
    #         # create_gmm_handle_f32(gmm32, <void*> _ws_ptr)
    #         setup_f32(gmm32)
    # if self.precision == 'double':
    #     with nogil:
    #         setup_f64(gmm64)
    #         # create_gmm_handle_f64(gmm64, <void*> _ws_ptr)

class _GaussianMixtureBackend :
    def __init__(self) :
        pass

    def step(self):
        cdef uintptr_t _dX_ptr = self.dX.device_ctypes_pointer.value

        cdef GMM[float] gmm32
        cdef GMM[double] gmm64
        _setup_gmm(self, gmm32, gmm64, True)

        if self.precision == 'single':
            with nogil:
                update_llhd_f32(<float*>_dX_ptr, gmm32)

            print("resp")
            print(self.resp_)
            with nogil:

                update_rhos_f32(gmm32, <float*> _dX_ptr)

            print(self.resp_)
            with nogil:

                update_pis_f32(gmm32)

            print("weights")
            print(self.weights_ )
            with nogil:

                update_mus_f32(<float*>_dX_ptr, gmm32)

            print("means")
            print(self.means_)
            with nogil:

                update_sigmas_f32(<float*>_dX_ptr, gmm32)

            print("covars")
            print(self.covars_)
            with nogil:

                compute_lbow_f32(gmm32)

        if self.precision == 'double':
            with nogil:
                update_llhd_f64(<double*>_dX_ptr, gmm64)

            print("resp")
            print(self.resp_)
            with nogil:
                update_rhos_f64(gmm64, <double*> _dX_ptr)

                update_pis_f64(gmm64)
                update_mus_f64(<double*>_dX_ptr, gmm64)
                update_sigmas_f64(<double*>_dX_ptr, gmm64)

                compute_lbow_f64(gmm64)

    def allocate_ws(self):
        self._workspaceSize = -1

        cdef GMM[float] gmm32
        cdef GMM[double] gmm64
        _setup_gmm(self, gmm32, gmm64, False)

        cuda_context = cuda.current_context()
        available_mem = cuda_context.get_memory_info().free
        print('available mem before allocation', to_mb(available_mem), "Mb")

        if self.precision == 'single':
            with nogil:
                workspace_size = get_workspace_size_f32(gmm32)
        if self.precision == 'double':
            with nogil:
                workspace_size = get_workspace_size_f64(gmm64)

        print("\n----------------")
        print('Workspace size', to_mb(workspace_size), "Mb")
        print("----------------\n")

        self.workspace = cuda.to_device(np.zeros(workspace_size, dtype=np.int8))

        # self.workspace = cuda.to_device(np.zeros(workspace_size, dtype=self.dtype))
        self._workspace_size = workspace_size

        cuda_context = cuda.current_context()
        available_mem = cuda_context.get_memory_info().free
        print('available mem after allocation', to_mb(available_mem), "Mb")

        print("workspace size", self._workspace_size)

    def init_step(self):
        cdef uintptr_t _dX_ptr = self.dX.device_ctypes_pointer.value

        cdef GMM[float] gmm32
        cdef GMM[double] gmm64
        _setup_gmm(self, gmm32, gmm64, True)

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



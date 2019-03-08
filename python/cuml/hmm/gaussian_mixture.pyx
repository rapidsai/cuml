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

cdef extern from "hmm/hmm_variables.h" :
    cdef cppclass GMM[T]:
        pass

cdef extern from "hmm/gmm_py.h" nogil:

    cdef void init_test()

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
                       int,
                       int,
                       int)

    cdef void setup_f32(GMM[float]&)
    cdef void update_rhos_f32(GMM[float]&, float*)
    cdef void update_mus_f32(float*, GMM[float]&)
    cdef void update_sigmas_f32(float*, GMM[float]&)
    cdef void update_pis_f32(GMM[float]&)

RUP_SIZE = 32

class GaussianMixture:

    def _get_ctype_ptr(self, obj):
        return obj.device_ctypes_pointer.value

    def _get_column_ptr(self, obj):
        return self._get_ctype_ptr(obj._column._data.to_gpu_array())

    def _get_dtype(self, precision):
        return {
            'single': np.float32,
            'double': np.float64,
        }[precision]

    def __init__(self, n_components, max_iter, precision='single', seed=False):
        self.precision = precision
        self.dtype = self._get_dtype(precision)

        self.n_components = n_components
        self.max_iter = max_iter

    def step(self):
        cdef uintptr_t _dX_ptr = self.dParams["x"].device_ctypes_pointer.value
        cdef uintptr_t _dmu_ptr = self.dParams["mus"].device_ctypes_pointer.value
        cdef uintptr_t _dsigma_ptr = self.dParams["sigmas"].device_ctypes_pointer.value
        cdef uintptr_t _dPis_ptr = self.dParams["pis"].device_ctypes_pointer.value
        cdef uintptr_t _dPis_inv_ptr = self.dParams["inv_pis"].device_ctypes_pointer.value
        cdef uintptr_t _dLlhd_ptr = self.dParams["llhd"].device_ctypes_pointer.value

        cdef int lddx = self.ldd["x"]
        cdef int lddmu = self.ldd["mus"]
        cdef int lddsigma = self.ldd["sigmas"]
        cdef int lddsigma_full = self.ldd["sigmas"] * self.nDim
        cdef int lddPis = self.ldd["pis"]
        cdef int lddLlhd =self.ldd["llhd"]

        cdef int nCl = self.nCl
        cdef int nDim = self.nDim
        cdef int nObs = self.nObs

        cdef GMM[float] gmm



        if self.precision == 'single':
            with nogil:
                init_f32(gmm,
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
                         <int> nCl,
                         <int> nDim,
                         <int> nObs)

                setup_f32(gmm)
                # update_rhos_f32(gmm, <float*> _dX_ptr)
                update_mus_f32(<float*>_dX_ptr, gmm)
                update_sigmas_f32(<float*>_dX_ptr, gmm)
                update_pis_f32(gmm)

    def _initialize_parameters(self, X):
        self.nObs = X.shape[0]
        self.nDim = X.shape[1]
        self.nCl = self.n_components

        self.ldd = {"x" : roundup(self.nDim, RUP_SIZE),
                    "mus" : roundup(self.nDim, RUP_SIZE),
                    "sigmas" : roundup(self.nDim, RUP_SIZE),
                     "llhd" : roundup(self.nCl, RUP_SIZE),
                    "pis" : roundup(self.nCl, RUP_SIZE),
                    "inv_pis" : roundup(self.nCl, RUP_SIZE)}

        params = sample_parameters(self.nDim, self.nCl, dt=self.dtype)
        params["llhd"] = sample_matrix(self.nCl, self.nObs, self.dtype, isColNorm=True)
        params['inv_pis'] = sample_matrix(1, self.nCl, self.dtype, isRowNorm=True)
        params["x"] = X.T.astype(self.dtype)

        params = align_parameters(params, self.ldd)
        params = flatten_parameters(params)
        params = cast_parameters(params ,self.dtype)

        self.dParams = dict(
            (key, cuda.to_device(params[key])) for key in self.ldd.keys())

    def fit(self, X):
        self._initialize_parameters(X)

        for _ in range(self.max_iter) :
            self.step()

    @property
    def means_(self):
        mus = self.dParams["mus"].copy_to_host()
        mus = deallign(mus, self.nDim, self.nCl, self.ldd["mus"])
        return mus.T

    @property
    def covariances_(self):
        sigmas = self.dParams["sigmas"].copy_to_host()
        sigmas = deallign(sigmas, self.nDim, self.nCl * self.nDim,
                          self.ldd["sigmas"])
        sigmas = sigmas.reshape((self.nDim, self.nDim, self.nCl), order="F")
        return np.swapaxes(sigmas, 0, 2)


    @property
    def weights_(self):
        pis = self.dParams["pis"].copy_to_host()
        pis = deallign(pis, self.nCl, 1, self.ldd["pis"])
        return pis
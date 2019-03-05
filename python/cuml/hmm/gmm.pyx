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


cdef extern from "hmm/hmm_variables.h" :
    cdef cppclass GMM[T]:
        pass

cdef extern from "hmm/gmm_py.h" nogil:

    cdef void init_f32(GMM[float]&,
                  float*, float*, float*, float*, float*,
                  int, int, int, int, int, int,
                  int, int, int)
    cdef void update_rhos_f32(float*, GMM[float]&)
    cdef void update_mus_f32(float*, GMM[float]&)
    cdef void update_sigmas_f32(float*, GMM[float]&)
    cdef void update_pis_f32(GMM[float]&)


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

    def __init__(self, precision='single', seed=False):

        self.precision = precision
        self.dtype = self._get_dtype(precision)

    def step(self):

        cdef uintptr_t _dX_ptr = self.dX.device_ctypes_pointer.value
        cdef uintptr_t _dmu_ptr = self.dmu.device_ctypes_pointer.value
        cdef uintptr_t _dsigma_ptr = self.dsigma.device_ctypes_pointer.value
        cdef uintptr_t _dPis_ptr = self.dPis.device_ctypes_pointer.value
        cdef uintptr_t _dPis_inv_ptr = self.dPis_inv.device_ctypes_pointer.value
        cdef uintptr_t _dLlhd_ptr = self.dLlhd.device_ctypes_pointer.value

        cdef GMM[float] gmm

        cdef int lddx
        cdef int lddmu
        cdef int lddsigma
        cdef int lddsigma_full
        cdef int lddPis
        cdef int lddLlhd
        cdef int nCl
        cdef int nDim
        cdef int nObs

        lddx =self.lddx
        lddmu =self.lddmu
        lddsigma =self.lddsigma
        lddsigma_full =self.lddsigma_full
        lddPis =self.lddPis
        lddLlhd =self.lddLlhd
        nCl =self.nCl
        nDim =self.nDim
        nObs =self.nObs

        if isinstance(z, cudf.Series):
            z_ptr = self._get_column_ptr(z)

        elif isinstance(z, np.ndarray):
            z_dev = cuda.to_device(z)
            z_ptr = z.device_ctypes_pointer.value

        elif cuda.devicearray.is_cuda_ndarray(z):
            z_ptr = z.device_ctypes_pointer.value

        else:
            msg = "z format supported"
            raise TypeError(msg)

        if self.precision == 'single':

            with nogil:
              init_f32(gmm, dmu, _dsigma_ptr, _dPis_ptr, _dPis_inv_ptr, _dLlhd_ptr,
                            lddx, _dmu_ptr, lddsigma, lddsigma_full, lddPis, lddLlhd, nCl, nDim, nObs)
              update_rhos_f32(_dX_ptr, gmm)
              update_mus_f32(_dX_ptr, gmm)
              update_sigmas_f32(_dX_ptr, gmm)
              update_pis_f32(gmm)

    def initialize():
      def zero_out(A, m, n, ldda):
          A[m:(ldda)] = 0.
          return A

      def rand_x(nDim, nObs, lddx):
          X = np.random.rand(lddx, nObs)
          return zero_out(X, nDim, nObs, lddx)


      def rand_mus(nDim, nObs, nCl, lddmu):
          mus = np.random.rand(lddmu, nCl)
          return zero_out(mus, nDim, lddmu)


      def rand_sigmas(nDim, nCl, lddsigma):
          s = np.random.rand(lddsigma * nDim, k)
          s = s + s.T
          return zero_out(s, nDim * lddsigma, lddsigma)


      def rand_llhd(nCl, nObs, ldLlhd):
          return np.random.rand(lddLlhd, nObs)

      self.mus = cuda.to_device(mus)
      self.sigmas = cuda.to_device(sigmas)
      self.rhos = cuda.to_device(rhos)

    def fit(self, X, nCl):
      self.dX = cuda.to_device(X)

      self.nCl = nCl
      self.nDim = X.shape[0]
      self.nObs = X.shape[1]

      for _ in range(self.n_iter) :
        self.step()

    def __setattr__(self, name, value):
        if name in ["z"]:
            if (isinstance(value, cudf.DataFrame)):
                val = numba_utils.row_matrix(value)

            elif (isinstance(value, cudf.Series)):
                val = value.to_gpu_array()

            elif (isinstance(value, np.ndarray) or cuda.devicearray.is_cuda_ndarray(value)):
                val = cuda.to_device(value)

            super(GMM, self).__setattr__(name, val)

        else:
            super(GMM, self).__setattr__(name, value)

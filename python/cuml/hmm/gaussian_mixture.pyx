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
from cuml.hmm.gmm_utils import roundup

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

        cdef int lddx = self.lddx
        cdef int lddmu = self.lddmu
        cdef int lddsigma = self.lddsigma
        cdef int lddsigma_full = self.lddsigma_full
        cdef int lddPis = self.lddPis
        cdef int lddLlhd =self.lddLlhd
        cdef int nCl = self.nCl
        cdef int nDim = self.nDim
        cdef int nObs = self.nObs

        cdef GMM[float] gmm

        if self.precision == 'single':
            with nogil:
              # TODO : Check pointers
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

    def initialize(self):
      mus = sample_mus(self.nDim, self.nCl, self.lddmu).astype(self.dtype)
      sigmas = sample_sigmas(self.nDim, self.nCl, self.lddsigma).astype(self.dtype)
      pis = sample_pis(self.nCl, self.lddPis).astype(self.dtype)
      llhd = sample_llhd(self.nCl, self.nObs, self.lddLlhd).astype(self.dtype)
      inv_pis = sample_pis(self.nCl, self.lddPis).astype(self.dtype)

      self.dmu = cuda.to_device(mus)
      self.dsigma = cuda.to_device(sigmas)
      self.dPis = cuda.to_device(pis)
      self.dPis_inv = cuda.to_device(inv_pis)
      self.dLlhd = cuda.to_device(llhd)

    def fit(self, X, nCl, nDim, nObs, n_iter):

      self.nCl = int(nCl)
      self.nDim = int(nDim)
      self.nObs = int(nObs)

      self.lddx = roundup(self.nDim, RUP_SIZE)
      self.lddmu = roundup(self.nDim, RUP_SIZE)
      self.lddsigma = roundup(self.nDim, RUP_SIZE)
      self.lddsigma_full = roundup(self.nDim * self.lddsigma, RUP_SIZE)
      self.lddLlhd = roundup(self.nCl, RUP_SIZE)
      self.lddPis = roundup(self.nCl, RUP_SIZE)

      self.dX = cuda.to_device(X.astype(self.dtype))

      self.initialize()

      for _ in range(n_iter) :
        self.step()

    def __setattr__(self, name, value):
        if name in ["dmu"]:
            if (isinstance(value, cudf.DataFrame)):
                val = numba_utils.row_matrix(value)

            elif (isinstance(value, cudf.Series)):
                val = value.to_gpu_array()

            elif (isinstance(value, np.ndarray) or cuda.devicearray.is_cuda_ndarray(value)):
                val = cuda.to_device(value)

            super(GaussianMixture, self).__setattr__(name, val)

        else:
            super(GaussianMixture, self).__setattr__(name, value)

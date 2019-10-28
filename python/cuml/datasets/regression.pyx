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

import cuml
import numpy as np

from cuml.common.handle cimport cumlHandle
from cuml.utils import get_dev_array_ptr, zeros

from libcpp cimport bool
from libc.stdint cimport uint64_t, uintptr_t

from random import randint

cdef extern from "cuml/datasets/make_regression.hpp" namespace "ML":
    void cpp_make_regression "ML::Datasets::make_regression" (
        const cumlHandle& handle,
        float* out,
        float* values,
        long n_rows,
        long n_cols,
        long n_informative,
        float* coef,
        long n_targets,
        float bias,
        long effective_rank,
        float tail_strength,
        float noise,
        bool shuffle,
        uint64_t seed)

    void cpp_make_regression "ML::Datasets::make_regression" (
        const cumlHandle& handle,
        double* out,
        double* values,
        long n_rows,
        long n_cols,
        long n_informative,
        double* coef,
        long n_targets,
        double bias,
        long effective_rank,
        double tail_strength,
        double noise,
        bool shuffle,
        uint64_t seed)

inp_to_dtype = {
    'single': np.float32,
    'float': np.float32,
    'double': np.float64,
    np.float32: np.float32,
    np.float64: np.float64
}

def make_regression(n_samples=100, n_features=2, n_informative=2, n_targets=1,
                    bias=0.0, effective_rank=None, tail_strength=0.5,
                    noise=0.0, shuffle=True, coef=False, random_state=None,
                    dtype='single', handle=None):
    """TODO: docs"""

    if dtype not in ['single', 'float', 'double', np.float32, np.float64]:
        raise TypeError("dtype must be either 'float' or 'double'")
    else:
        dtype = inp_to_dtype[dtype]

    if effective_rank is None:
        effective_rank = -1

    handle = cuml.common.handle.Handle() if handle is None else handle
    cdef cumlHandle* handle_ = <cumlHandle*><size_t>handle.getHandle()

    out = zeros((n_samples, n_features), dtype=dtype, order='C')
    cdef uintptr_t out_ptr = get_dev_array_ptr(out)

    values = zeros((n_samples, n_targets), dtype=dtype, order='C')
    cdef uintptr_t values_ptr = get_dev_array_ptr(values)

    cdef uintptr_t coef_ptr
    coef_ptr = <uintptr_t> NULL
    if coef:
        coefs = zeros((n_features, n_targets), dtype=dtype, order='C')
        coef_ptr = get_dev_array_ptr(coefs)

    if random_state is None:
        random_state = randint(0, 1e18)

    if dtype == np.float32:
        cpp_make_regression(handle_[0], <float*> out_ptr,
                            <float*> values_ptr, <long> n_samples,
                            <long> n_features, <long> n_informative,
                            <float*> coef_ptr, <long> n_targets, <float> bias,
                            <long> effective_rank, <float> tail_strength,
                            <float> noise, <bool> shuffle,
                            <uint64_t> random_state)

    else:
        cpp_make_regression(handle_[0], <double*> out_ptr,
                            <double*> values_ptr, <long> n_samples,
                            <long> n_features, <long> n_informative,
                            <double*> coef_ptr, <long> n_targets,
                            <double> bias, <long> effective_rank,
                            <double> tail_strength, <double> noise,
                            <bool> shuffle, <uint64_t> random_state)

    if coef:
        return out, values, coefs
    else:
        return out, values

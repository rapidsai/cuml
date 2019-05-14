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

from cuml.common.handle cimport cumlHandle
import cuml.common.handle
from libc.stdint cimport uintptr_t

from cuml.metrics cimport regression


def r2_score(y, y_hat, handle=None):
    handle = cuml.common.handle.Handle() if handle is None else handle
    cdef cumlHandle* handle_ = <cumlHandle*><size_t>handle.getHandle()

    cdef uintptr_t y_ptr = y.device_ctypes_pointer.value
    cdef uintptr_t y_hat_ptr = y_hat.device_ctypes_pointer.value

    cdef float result_f32
    cdef double result_f64

    n = len(y)

    if y.dtype == 'float32':

        result_f32 = regression.r2_score_py(handle_[0],
                            <float*> y_ptr,
                            <float*> y_hat_ptr,
                            <int> n)

        result = result_f32

    else:
        result_f64 = regression.r2_score_py(handle_[0],
                            <double*> y_ptr,
                            <double*> y_hat_ptr,
                            <int> n)

        result = result_f64


    return result







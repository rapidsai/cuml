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

from libc.stdint cimport uintptr_t

cdef extern from "metrics/trustworthiness.h" namespace "ML":
    cdef double cuml_trustworthiness[T](T* X, T* X_embedded, int n, int m, int d, int n_neighbors)


"""
Expresses to what extent the local structure is retained in embedding.
The score is defined in the range [0, 1].

Parameters
----------
    X : cuDF DataFrame or Numpy array (n_samples, n_features)
        Data in original dimension

    X : cuDF DataFrame or Numpy array (n_samples, n_components)
        Data in target dimension (embedding)

    n_neighbors : int, optional (default: 5)
        Number of neighbors considered

Returns
-------
    trustworthiness score : double
        Trustworthiness of the low-dimensional embedding
"""
def trustworthiness(X, X_embedded, n_neighbors=5):
    n, m = X.shape
    d = X_embedded.shape[1]

    if X.dtype != X_embedded.dtype:
        raise TypeError("X and X_embedded parameters must be of same type")

    if X.dtype != np.float32 or X_embedded.dtype != np.float32: # currently only float32 is available
        return TypeError("X and X_embedded parameters must be of type float32")

    cdef uintptr_t d_X = get_ctype_ptr(cuda.to_device(X))
    cdef uintptr_t d_X_embedded = get_ctype_ptr(cuda.to_device(X_embedded))

    if X.dtype == np.float32:
        return cuml_trustworthiness[float](<float*>d_X, <float*>d_X_embedded, n, m, d, n_neighbors)
    #else:
    #    return cuml_trustworthiness(<double*>d_X, <double*>d_X_embedded, n, m, d, n_neighbors)



def get_ctype_ptr(obj):
        # The manner to access the pointers in the gdf's might change, so
        # encapsulating access in the following 3 methods. They might also be
        # part of future gdf versions.
        return obj.device_ctypes_pointer.value
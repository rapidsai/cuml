#
# Copyright (c) 2018-2019, NVIDIA CORPORATION.
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
import warnings

from numba import cuda

from libc.stdint cimport uintptr_t
from cuml.common.handle cimport cumlHandle
from cuml.utils import get_cudf_column_ptr, get_dev_array_ptr, \
    input_to_dev_array

cdef extern from "metrics/trustworthiness_c.h" namespace "MLCommon::Distance":

    ctypedef int DistanceType
    ctypedef DistanceType euclidean "(MLCommon::Distance::DistanceType)5"

cdef extern from "metrics/trustworthiness_c.h" namespace "ML::Metrics":

    cdef double trustworthiness_score[T, DistanceType](const cumlHandle& h,
                                                       T* X,
                                                       T* X_embedded,
                                                       int n, int m,
                                                       int d,
                                                       int n_neighbors)


def _get_array_ptr(obj):
    """
    Get ctype pointer of a numba style device array
    """
    return obj.device_ctypes_pointer.value


def trustworthiness(X, X_embedded, handle=None, n_neighbors=5,
                    metric='euclidean', should_downcast=True,
                    convert_dtype=False):
    """
    Expresses to what extent the local structure is retained in embedding.
    The score is defined in the range [0, 1].

    Parameters
    ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        X_embedded : array-like (device or host) shape= (n_samples, n_features)
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        n_neighbors : int, optional (default: 5)
            Number of neighbors considered

        convert_dtype : bool, optional (default = False)
            When set to True, the trustworthiness method will automatically
            convert the inputs to np.float32.

    Returns
    -------
        trustworthiness score : double
            Trustworthiness of the low-dimensional embedding
    """

    if should_downcast:
        convert_dtype = True
        warnings.warn("Parameter should_downcast is deprecated, use "
                      "convert_dtype instead. ")

    cdef uintptr_t d_X_ptr
    cdef uintptr_t d_X_embedded_ptr

    X_m, d_X_ptr, n_samples, n_features, dtype1 = \
        input_to_dev_array(X, order='C', check_dtype=np.float32,
                           convert_to_dtype=(np.float32 if convert_dtype
                                             else None))
    X_m2, d_X_embedded_ptr, n_rows, n_components, dtype2 = \
        input_to_dev_array(X_embedded, order='C',
                           check_dtype=np.float32,
                           convert_to_dtype=(np.float32 if convert_dtype
                                             else None))

    cdef cumlHandle* handle_ = <cumlHandle*>0
    if handle is None:
        handle_ = <cumlHandle*><size_t>(new cumlHandle())
    else:
        handle_ = <cumlHandle*><size_t>handle.getHandle()

    if metric == 'euclidean':
        res = trustworthiness_score[float, euclidean](handle_[0],
                                                      <float*>d_X_ptr,
                                                      <float*>d_X_embedded_ptr,
                                                      n_samples,
                                                      n_features,
                                                      n_components,
                                                      n_neighbors)
        del X_m
        del X_m2
    else:
        del X_m
        del X_m2
        raise Exception("Unknown metric")

    if handle is None:
        del handle_
    return res

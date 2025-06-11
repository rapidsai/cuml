#
# Copyright (c) 2018-2025, NVIDIA CORPORATION.
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

# distutils: language = c++

import numpy as np
from pylibraft.common.handle import Handle

import cuml.internals
from cuml.internals.input_utils import input_to_cuml_array

from libc.stdint cimport uintptr_t
from pylibraft.common.handle cimport handle_t

from cuml.metrics.distance_type cimport DistanceType


cdef extern from "cuml/common/distance_type.hpp" namespace "ML::distance" nogil:
    ctypedef DistanceType euclidean "(ML::distance::DistanceType)5"

cdef extern from "cuml/metrics/metrics.hpp" namespace "ML::Metrics" nogil:

    cdef double trustworthiness_score[T, DistanceType](const handle_t& h,
                                                       T* X,
                                                       T* X_embedded,
                                                       int n, int m,
                                                       int d,
                                                       int n_neighbors,
                                                       int batchSize) \
        except +


def _get_array_ptr(obj):
    """
    Get ctype pointer of a numba style device array
    """
    return obj.device_ctypes_pointer.value


@cuml.internals.api_return_any()
def trustworthiness(X, X_embedded, handle=None, n_neighbors=5,
                    metric='euclidean',
                    convert_dtype=True, batch_size=512) -> float:
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

        n_neighbors : int, optional (default=5)
            Number of neighbors considered

        metric : str in ['euclidean'] (default='euclidean')
            Metric used to compute the trustworthiness. For the moment only
            'euclidean' is supported.

        convert_dtype : bool, optional (default=False)
            When set to True, the trustworthiness method will automatically
            convert the inputs to np.float32.

        batch_size : int (default=512)
            The number of samples to use for each batch.

    Returns
    -------
        trustworthiness score : double
            Trustworthiness of the low-dimensional embedding
    """

    if n_neighbors > X.shape[0]:
        raise ValueError("n_neighbors must be <= the number of rows.")

    if n_neighbors > X.shape[0]:
        raise ValueError("n_neighbors must be <= the number of rows.")

    handle = Handle() if handle is None else handle

    cdef uintptr_t d_X_ptr
    cdef uintptr_t d_X_embedded_ptr

    X_m, n_samples, n_features, _ = \
        input_to_cuml_array(X, order='C', check_dtype=np.float32,
                            convert_to_dtype=(np.float32 if convert_dtype
                                              else None))
    d_X_ptr = X_m.ptr

    X_m2, _, n_components, _ = \
        input_to_cuml_array(X_embedded, order='C',
                            check_dtype=np.float32,
                            convert_to_dtype=(np.float32 if convert_dtype
                                              else None))
    d_X_embedded_ptr = X_m2.ptr

    handle = Handle() if handle is None else handle
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

    if metric == 'euclidean':
        ret = trustworthiness_score[float, euclidean](handle_[0],
                                                      <float*>d_X_ptr,
                                                      <float*>d_X_embedded_ptr,
                                                      n_samples,
                                                      n_features,
                                                      n_components,
                                                      n_neighbors,
                                                      batch_size)
        handle.sync()

    else:
        raise Exception("Unknown metric")

    return ret

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

from numba import cuda

from libc.stdint cimport uintptr_t
from cuml.common.handle cimport cumlHandle

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
                    metric='euclidean', should_downcast=True):
    """
    Expresses to what extent the local structure is retained in embedding.
    The score is defined in the range [0, 1].

    Parameters
    ----------
        X : cuDF DataFrame or Numpy array (n_samples, n_features)
            Data in original dimension

        X_embedded : cuDF DataFrame or Numpy array (n_samples, n_components)
            Data in target dimension (embedding)

        n_neighbors : int, optional (default: 5)
            Number of neighbors considered

    Returns
    -------
        trustworthiness score : double
            Trustworthiness of the low-dimensional embedding
    """
    if (isinstance(X, cudf.DataFrame) and
            isinstance(X_embedded, cudf.DataFrame)):
        datatype1 = np.dtype(X[X.columns[0]]._column.dtype)
        datatype2 = np.dtype(X_embedded[X_embedded.columns[0]]._column.dtype)
        n_samples = len(X)
        n_features = len(X._cols)
        n_components = len(X_embedded._cols)
    elif isinstance(X, np.ndarray) and isinstance(X_embedded, np.ndarray):
        datatype1 = X.dtype
        datatype2 = X_embedded.dtype
        n_samples, n_features = X.shape
        n_components = X_embedded.shape[1]
    else:
        raise TypeError("X and X_embedded parameters must both be cuDF"
                        " Dataframes or Numpy ndarray")

    if datatype1 != np.float32 or datatype2 != np.float32:
        if should_downcast:
            X = to_single_precision(X)
            X_embedded = to_single_precision(X_embedded)
        else:
            raise Exception("Input is double precision. Use "
                            "'should_downcast=True' "
                            "if you'd like it to be automatically "
                            "casted to single precision.")

    if isinstance(X, cudf.DataFrame):
        d_X = X.as_gpu_matrix(order='C')
        d_X_embedded = X_embedded.as_gpu_matrix(order='C')
    elif isinstance(X, np.ndarray):
        d_X = cuda.to_device(X)
        d_X_embedded = cuda.to_device(X_embedded)

    cdef uintptr_t d_X_ptr = _get_array_ptr(d_X)
    cdef uintptr_t d_X_embedded_ptr = _get_array_ptr(d_X_embedded)

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
    else:
        raise Exception("Unknown metric")

    if handle is None:
        del handle_
    return res


def to_single_precision(X):
    if isinstance(X, cudf.DataFrame):
        new_cols = [(col, X._cols[col].astype(np.float32)) for col in X._cols]
        overflowed = sum([len(colval[colval >= np.inf]) for colname, colval
                         in new_cols])

        if overflowed > 0:
            raise Exception("Downcast to single-precision resulted in data"
                            " loss.")

        X = cudf.DataFrame(new_cols)
    else:
        X = X.astype(np.float32)
        overflowed = len(X[X >= np.inf])

        if overflowed > 0:
            raise Exception("Downcast to single-precision resulted in data"
                            " loss.")

    return X

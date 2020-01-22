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

from cuml.neighbors.nearest_neighbors import NearestNeighbors

from cuml.utils import get_cudf_column_ptr, get_dev_array_ptr, \
    input_to_dev_array, zeros, row_matrix

from cuml.metrics import r2_score

import numpy as np

import cudf

from cython.operator cimport dereference as deref

from libcpp.vector cimport vector

from cuml.common.handle cimport cumlHandle

from libcpp cimport bool
from libcpp.memory cimport shared_ptr

import rmm
from libc.stdlib cimport malloc, free

from libc.stdint cimport uintptr_t, int64_t
from libc.stdlib cimport calloc, malloc, free

from numba import cuda
import rmm

cimport cuml.common.handle
cimport cuml.common.cuda

cdef extern from "cuml/cuml.hpp" namespace "ML" nogil:
    cdef cppclass deviceAllocator:
        pass

    cdef cppclass cumlHandle:
        cumlHandle() except +
        void setStream(cuml.common.cuda._Stream s) except +
        void setDeviceAllocator(shared_ptr[deviceAllocator] a) except +
        cuml.common.cuda._Stream getStream() except +

cdef extern from "cuml/neighbors/knn.hpp" namespace "ML":

    void knn_regress(
        cumlHandle &handle,
        float *out,
        int64_t *knn_indices,
        vector[float *] &y,
        size_t n_samples,
        int k,
    ) except +


class KNeighborsRegressor(NearestNeighbors):
    """

    K-Nearest Neighbors Regressor is an instance-based learning technique,
    that keeps training samples around for prediction, rather than trying
    to learn a generalizable set of model parameters.

    The K-Nearest Neighbors Regressor will compute the average of the
    labels for the k closest neighbors and use it as the label.

    Parameters
    ----------
    n_neighbors : int (default=5)
        Default number of neighbors to query
    verbose : boolean (default=False)
        Whether to print verbose logs
    handle : cumlHandle
        The cumlHandle resources to use
    algorithm : string (default='brute')
        The query algorithm to use. Currently, only 'brute' is supported.
    metric : string (default='euclidean').
        Distance metric to use.
    weights : string (default='uniform')
        Sample weights to use. Currently, only the uniform strategy is
        supported.

    Examples
    ---------
    .. code-block:: python

      from cuml.neighbors import KNeighborsRegressor

      from sklearn.datasets import make_blobs
      from sklearn.model_selection import train_test_split

      X, y = make_blobs(n_samples=100, centers=5,
                        n_features=10)

      knn = KNeighborsRegressor(n_neighbors=10)

      X_train, X_test, y_train, y_test =
        train_test_split(X, y, train_size=0.80)

      knn.fit(X_train, y_train)

      knn.predict(X_test)


    Output:


    .. code-block:: python

      array([3.        , 1.        , 1.        , 3.79999995, 2.        ,
             0.        , 3.79999995, 3.79999995, 3.79999995, 0.        ,
             3.79999995, 0.        , 1.        , 2.        , 3.        ,
             1.        , 0.        , 0.        , 0.        , 2.        ,
             3.        , 3.        , 0.        , 3.        , 3.79999995,
             3.79999995, 3.79999995, 3.79999995, 3.        , 2.        ,
             3.79999995, 3.79999995, 0.        ])



    Notes
    ------

    For additional docs, see `scikitlearn's KNeighborsClassifier
    <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>`_.
    """

    def __init__(self, weights="uniform", **kwargs):
        super(KNeighborsRegressor, self).__init__(**kwargs)
        self.y = None
        self.weights = weights
        if weights != "uniform":
            raise ValueError("Only uniform weighting strategy "
                             "is supported currently.")

    def fit(self, X, y, convert_dtype=True):
        """
        Fit a GPU index for k-nearest neighbors regression model.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        y : array-like (device or host) shape = (n_samples, n_outputs)
            Dense matrix (floats or doubles) of shape (n_samples, n_outputs).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        convert_dtype : bool, optional (default = True)
            When set to True, the fit method will automatically
            convert the inputs to np.float32.
        """
        super(KNeighborsRegressor, self).fit(X, convert_dtype=convert_dtype)
        self.y, _, _, _, _ = \
            input_to_dev_array(y, order='F', check_dtype=np.float32,
                               convert_to_dtype=(np.float32
                                                 if convert_dtype
                                                 else None))

        self.handle.sync()

    def predict(self, X, convert_dtype=True):
        """
        Use the trained k-nearest neighbors regression model to
        predict the labels for X

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        convert_dtype : bool, optional (default = True)
            When set to True, the fit method will automatically
            convert the inputs to np.float32.
        """
        knn_indices = self.kneighbors(X, return_distance=False,
                                      convert_dtype=convert_dtype)

        cdef uintptr_t inds_ctype

        inds, inds_ctype, n_rows, n_cols, dtype = \
            input_to_dev_array(knn_indices, order='C', check_dtype=np.int64,
                               convert_to_dtype=(np.int64
                                                 if convert_dtype
                                                 else None))

        res_cols = 1 if len(self.y.shape) == 1 else self.y.shape[1]
        res_shape = n_rows if res_cols == 1 else (n_rows, res_cols)
        results = rmm.to_device(zeros(res_shape, dtype=np.float32,
                                      order="C"))

        cdef uintptr_t results_ptr = get_dev_array_ptr(results)
        cdef uintptr_t y_ptr
        cdef vector[float*] *y_vec = new vector[float*]()

        for col_num in range(res_cols):
            col = self.y if res_cols == 1 else self.y[:, col_num]
            y_ptr = get_dev_array_ptr(col)
            y_vec.push_back(<float*>y_ptr)

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        knn_regress(
            handle_[0],
            <float*>results_ptr,
            <int64_t*>inds_ctype,
            deref(y_vec),
            <size_t>X.shape[0],
            <int>self.n_neighbors
        )

        self.handle.sync()
        if isinstance(X, np.ndarray):
            return np.array(results)
        elif isinstance(X, cudf.DataFrame):
            if results.ndim == 1:
                results = results.reshape(results.shape[0], 1)
            return cudf.DataFrame.from_gpu_matrix(results)
        else:
            return results

    def score(self, X, y, convert_dtype=True):
        """
        Fit a GPU index for k-nearest neighbors regression model.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        y : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        convert_dtype : bool, optional (default = True)
            When set to True, the fit method will automatically
            convert the inputs to np.float32.
        """
        y_hat = self.predict(X, convert_dtype=convert_dtype)
        return r2_score(y, y_hat, convert_dtype=convert_dtype)

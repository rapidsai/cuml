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

from cuml.common.array import CumlArray
from cuml.common import input_to_cuml_array
from cuml.common.base import ClassifierMixin

import numpy as np
import cupy as cp

import cudf

from cython.operator cimport dereference as deref

from cuml.common.handle cimport cumlHandle
from libcpp.vector cimport vector

from cuml.common import with_cupy_rmm

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


cdef extern from "cuml/neighbors/knn.hpp" namespace "ML":

    void knn_classify(
        cumlHandle &handle,
        int* out,
        int64_t *knn_indices,
        vector[int*] &y,
        size_t n_index_rows,
        size_t n_samples,
        int k
    ) except +

    void knn_class_proba(
        cumlHandle &handle,
        vector[float*] &out,
        int64_t *knn_indices,
        vector[int*] &y,
        size_t n_index_rows,
        size_t n_samples,
        int k
    ) except +


class KNeighborsClassifier(NearestNeighbors, ClassifierMixin):
    """
    K-Nearest Neighbors Classifier is an instance-based learning technique,
    that keeps training samples around for prediction, rather than trying
    to learn a generalizable set of model parameters.

    Parameters
    ----------
    n_neighbors : int (default=5)
        Default number of neighbors to query
    verbose : int or boolean (default = False)
        Logging level
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

      from cuml.neighbors import KNeighborsClassifier

      from sklearn.datasets import make_blobs
      from sklearn.model_selection import train_test_split

      X, y = make_blobs(n_samples=100, centers=5,
                        n_features=10)

      knn = KNeighborsClassifier(n_neighbors=10)

      X_train, X_test, y_train, y_test =
        train_test_split(X, y, train_size=0.80)

      knn.fit(X_train, y_train)

      knn.predict(X_test)


    Output:
    -------

    .. code-block:: python

      array([3, 1, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 1, 0, 0, 0, 2, 3, 3,
             0, 3, 0, 0, 0, 0, 3, 2, 0, 0, 0], dtype=int32)

    Notes
    ------

    For additional docs, see `scikitlearn's KNeighborsClassifier
    <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>`_.
    """

    def __init__(self, weights="uniform", **kwargs):
        super(KNeighborsClassifier, self).__init__(**kwargs)

        self.y = None
        self.weights = weights

        if weights != "uniform":
            raise ValueError("Only uniform weighting strategy is "
                             "supported currently.")

    def fit(self, X, y, convert_dtype=True):
        """
        Fit a GPU index for k-nearest neighbors classifier model.

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
        self._set_target_dtype(y)

        super(KNeighborsClassifier, self).fit(X, convert_dtype)
        self.y, _, _, _ = \
            input_to_cuml_array(y, order='F', check_dtype=np.int32,
                                convert_to_dtype=(np.int32
                                                  if convert_dtype
                                                  else None))
        return self

    def predict(self, X, convert_dtype=True):
        """
        Use the trained k-nearest neighbors classifier to
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

        Returns
        ----------
        y : (same as the input datatype)
            Dense vector (ints, floats, or doubles) of shape (n_samples, 1).
        """

        out_type = self._get_output_type(X)
        out_dtype = self._get_target_dtype()

        knn_indices = self.kneighbors(X, return_distance=False,
                                      convert_dtype=convert_dtype)

        inds, n_rows, _, _ = \
            input_to_cuml_array(knn_indices, order='C', check_dtype=np.int64,
                                convert_to_dtype=(np.int64
                                                  if convert_dtype
                                                  else None))
        cdef uintptr_t inds_ctype = inds.ptr

        out_cols = self.y.shape[1] if len(self.y.shape) == 2 else 1

        out_shape = (n_rows, out_cols) if out_cols > 1 else n_rows

        classes = CumlArray.zeros(out_shape, dtype=np.int32, order="C")

        cdef vector[int*] *y_vec = new vector[int*]()

        # If necessary, separate columns of y to support multilabel
        # classification
        cdef uintptr_t y_ptr
        for i in range(out_cols):
            col = self.y[:, i] if out_cols > 1 else self.y
            y_ptr = col.ptr
            y_vec.push_back(<int*>y_ptr)

        cdef uintptr_t classes_ptr = classes.ptr

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        knn_classify(
            handle_[0],
            <int*> classes_ptr,
            <int64_t*>inds_ctype,
            deref(y_vec),
            <size_t>self.n_rows,
            <size_t>n_rows,
            <int>self.n_neighbors
        )

        self.handle.sync()

        return classes.to_output(output_type=out_type, output_dtype=out_dtype)

    @with_cupy_rmm
    def predict_proba(self, X, convert_dtype=True):
        """
        Use the trained k-nearest neighbors classifier to
        predict the label probabilities for X

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

        out_type = self._get_output_type(X)

        knn_indices = self.kneighbors(X, return_distance=False,
                                      convert_dtype=convert_dtype)

        inds, n_rows, n_cols, dtype = \
            input_to_cuml_array(knn_indices, order='C',
                                check_dtype=np.int64,
                                convert_to_dtype=(np.int64
                                                  if convert_dtype
                                                  else None))
        cdef uintptr_t inds_ctype = inds.ptr

        out_cols = self.y.shape[1] if len(self.y.shape) == 2 else 1

        cdef vector[int*] *y_vec = new vector[int*]()
        cdef vector[float*] *out_vec = new vector[float*]()

        out_classes = []
        cdef uintptr_t classes_ptr
        cdef uintptr_t y_ptr
        for out_col in range(out_cols):
            col = self.y[:, out_col] if out_cols > 1 else self.y
            classes = CumlArray.zeros((n_rows,
                                       len(cp.unique(cp.asarray(col)))),
                                      dtype=np.float32,
                                      order="C")
            out_classes.append(classes)
            classes_ptr = classes.ptr
            out_vec.push_back(<float*>classes_ptr)

            y_ptr = col.ptr
            y_vec.push_back(<int*>y_ptr)

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()

        knn_class_proba(
            handle_[0],
            deref(out_vec),
            <int64_t*>inds_ctype,
            deref(y_vec),
            <size_t>self.n_rows,
            <size_t>n_rows,
            <int>self.n_neighbors
        )

        self.handle.sync()

        final_classes = []
        for out_class in out_classes:
            final_classes.append(out_class.to_output(out_type))

        return final_classes[0] \
            if len(final_classes) == 1 else tuple(final_classes)

    def get_param_names(self):
        return super(KNeighborsClassifier, self).get_param_names()\
            + ["weights"]

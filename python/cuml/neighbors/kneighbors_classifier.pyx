#
# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

import typing

from cuml.neighbors.nearest_neighbors import NearestNeighbors

import cuml.internals
from cuml.internals.array import CumlArray
from cuml.common import input_to_cuml_array
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.internals.mixins import ClassifierMixin
from cuml.common.doc_utils import generate_docstring
from cuml.internals.mixins import FMajorInputTagMixin

import numpy as np
import cupy as cp


from cython.operator cimport dereference as deref

from pylibraft.common.handle cimport handle_t
from libcpp.vector cimport vector

from libcpp cimport bool
from libcpp.memory cimport shared_ptr

import rmm
from libc.stdlib cimport malloc, free

from libc.stdint cimport uintptr_t, int64_t
from libc.stdlib cimport calloc, malloc, free

from numba import cuda
import rmm

cimport cuml.common.cuda


cdef extern from "cuml/neighbors/knn.hpp" namespace "ML":

    void knn_classify(
        handle_t &handle,
        int* out,
        int64_t *knn_indices,
        vector[int*] &y,
        size_t n_index_rows,
        size_t n_samples,
        int k
    ) except +

    void knn_class_proba(
        handle_t &handle,
        vector[float*] &out,
        int64_t *knn_indices,
        vector[int*] &y,
        size_t n_index_rows,
        size_t n_samples,
        int k
    ) except +


class KNeighborsClassifier(ClassifierMixin,
                           FMajorInputTagMixin,
                           NearestNeighbors):
    """
    K-Nearest Neighbors Classifier is an instance-based learning technique,
    that keeps training samples around for prediction, rather than trying
    to learn a generalizable set of model parameters.

    Parameters
    ----------
    n_neighbors : int (default=5)
        Default number of neighbors to query
    algorithm : string (default='auto')
        The query algorithm to use. Currently, only 'brute' is supported.
    metric : string (default='euclidean').
        Distance metric to use.
    weights : string (default='uniform')
        Sample weights to use. Currently, only the uniform strategy is
        supported.
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Examples
    --------

    .. code-block:: python

        >>> from cuml.neighbors import KNeighborsClassifier
        >>> from cuml.datasets import make_blobs
        >>> from cuml.model_selection import train_test_split

        >>> X, y = make_blobs(n_samples=100, centers=5,
        ...                   n_features=10, random_state=5)
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...     X, y, train_size=0.80, random_state=5)

        >>> knn = KNeighborsClassifier(n_neighbors=10)

        >>> knn.fit(X_train, y_train)
        KNeighborsClassifier()
        >>> knn.predict(X_test) # doctest: +SKIP
        array([1., 2., 2., 3., 4., 2., 4., 4., 2., 3., 1., 4., 3., 1., 3., 4., 3., # noqa: E501
            4., 1., 3.], dtype=float32)

    Notes
    -----

    For additional docs, see `scikitlearn's KNeighborsClassifier
    <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>`_.
    """

    y = CumlArrayDescriptor()
    classes_ = CumlArrayDescriptor()

    def __init__(self, *, weights="uniform", handle=None, verbose=False,
                 output_type=None, **kwargs):
        super().__init__(
            handle=handle,
            verbose=verbose,
            output_type=output_type,
            **kwargs)

        self.y = None
        self.classes_ = None
        self.weights = weights

        if weights != "uniform":
            raise ValueError("Only uniform weighting strategy is "
                             "supported currently.")

    @generate_docstring(convert_dtype_cast='np.float32')
    @cuml.internals.api_base_return_any(set_output_dtype=True)
    def fit(self, X, y, convert_dtype=True) -> "KNeighborsClassifier":
        """
        Fit a GPU index for k-nearest neighbors classifier model.

        """
        super(KNeighborsClassifier, self).fit(X, convert_dtype)
        self.y, _, _, _ = \
            input_to_cuml_array(y, order='F', check_dtype=np.int32,
                                convert_to_dtype=(np.int32
                                                  if convert_dtype
                                                  else None))
        self.classes_ = cp.unique(self.y)
        return self

    @generate_docstring(convert_dtype_cast='np.float32',
                        return_values={'name': 'X_new',
                                       'type': 'dense',
                                       'description': 'Labels predicted',
                                       'shape': '(n_samples, 1)'})
    @cuml.internals.api_base_return_array(get_output_dtype=True)
    def predict(self, X, convert_dtype=True) -> CumlArray:
        """
        Use the trained k-nearest neighbors classifier to
        predict the labels for X

        """
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

        classes = CumlArray.zeros(out_shape, dtype=np.int32, order="C",
                                  index=inds.index)

        cdef vector[int*] *y_vec = new vector[int*]()

        # If necessary, separate columns of y to support multilabel
        # classification
        cdef uintptr_t y_ptr
        for i in range(out_cols):
            col = self.y[:, i] if out_cols > 1 else self.y
            y_ptr = col.ptr
            y_vec.push_back(<int*>y_ptr)

        cdef uintptr_t classes_ptr = classes.ptr

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        knn_classify(
            handle_[0],
            <int*> classes_ptr,
            <int64_t*>inds_ctype,
            deref(y_vec),
            <size_t>self.n_samples_fit_,
            <size_t>n_rows,
            <int>self.n_neighbors
        )

        self.handle.sync()

        return classes

    @generate_docstring(convert_dtype_cast='np.float32',
                        return_values={'name': 'X_new',
                                       'type': 'dense',
                                       'description': 'Labels probabilities',
                                       'shape': '(n_samples, 1)'})
    @cuml.internals.api_base_return_generic()
    def predict_proba(
            self,
            X,
            convert_dtype=True) -> typing.Union[CumlArray, typing.Tuple]:
        """
        Use the trained k-nearest neighbors classifier to
        predict the label probabilities for X

        """
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
                                      order="C",
                                      index=inds.index)
            out_classes.append(classes)
            classes_ptr = classes.ptr
            out_vec.push_back(<float*>classes_ptr)

            y_ptr = col.ptr
            y_vec.push_back(<int*>y_ptr)

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        knn_class_proba(
            handle_[0],
            deref(out_vec),
            <int64_t*>inds_ctype,
            deref(y_vec),
            <size_t>self.n_samples_fit_,
            <size_t>n_rows,
            <int>self.n_neighbors
        )

        self.handle.sync()

        final_classes = []
        for out_class in out_classes:
            final_classes.append(out_class)

        return final_classes[0] \
            if len(final_classes) == 1 else tuple(final_classes)

    def get_param_names(self):
        return super().get_param_names() + ["weights"]

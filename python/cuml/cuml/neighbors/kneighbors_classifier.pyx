#
# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
from __future__ import annotations

import cupy as cp
import numpy as np

import cuml
from cuml.common import input_to_cuml_array
from cuml.common.classification import decode_labels, preprocess_labels
from cuml.common.doc_utils import generate_docstring
from cuml.internals import get_handle
from cuml.internals.array import CumlArray
from cuml.internals.interop import UnsupportedOnGPU
from cuml.internals.mixins import ClassifierMixin, FMajorInputTagMixin
from cuml.internals.outputs import reflect, run_in_internal_context
from cuml.neighbors.nearest_neighbors import NearestNeighbors
from cuml.neighbors.weights import compute_weights

from libc.stdint cimport int64_t, uintptr_t
from libcpp.vector cimport vector
from pylibraft.common.handle cimport handle_t


cdef extern from "cuml/neighbors/knn.hpp" namespace "ML" nogil:

    void knn_classify(
        handle_t &handle,
        int* out,
        int64_t *knn_indices,
        vector[int*] &y,
        size_t n_index_rows,
        size_t n_samples,
        int k,
        float *sample_weight
    ) except +

    void knn_class_proba(
        handle_t &handle,
        vector[float*] &out,
        int64_t *knn_indices,
        vector[int*] &y,
        size_t n_index_rows,
        size_t n_samples,
        int k,
        float *sample_weight
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
    weights : {'uniform', 'distance'} or callable, default='uniform'
        Weight function used in prediction. Possible values:

        - 'uniform' : uniform weights. All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          In this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.
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
    _cpu_class_path = "sklearn.neighbors.KNeighborsClassifier"

    @classmethod
    def _get_param_names(cls):
        return [*super()._get_param_names(), "weights"]

    @classmethod
    def _params_from_cpu(cls, model):
        if callable(model.weights):
            raise UnsupportedOnGPU(
                "Callable weights are not supported for CPU model conversion"
            )

        return {
            "weights": model.weights,
            **super()._params_from_cpu(model),
        }

    def _params_to_cpu(self):
        return {
            "weights": self.weights,
            **super()._params_to_cpu(),
        }

    def _attrs_from_cpu(self, model):
        return {
            "classes_": model.classes_,
            "_y": cp.asarray(model._y, order="F", dtype=np.int32),
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        return {
            "classes_": self.classes_,
            "_y": self._y.get(),
            "outputs_2d_": self.outputs_2d_,
            **super()._attrs_to_cpu(model),
        }

    def __init__(
        self,
        *,
        weights="uniform",
        handle=None,
        verbose=False,
        output_type=None,
        **kwargs,
    ):
        super().__init__(
            handle=handle, verbose=verbose, output_type=output_type, **kwargs
        )
        self.weights = weights

    @generate_docstring(convert_dtype_cast='np.float32')
    @reflect(reset=True)
    def fit(self, X, y, *, convert_dtype=True) -> "KNeighborsClassifier":
        """
        Fit a GPU index for k-nearest neighbors classifier model.

        """
        if self.weights not in ('uniform', 'distance', None) and not callable(self.weights):
            raise ValueError(
                f"weights must be 'uniform', 'distance', or a callable, got {self.weights}"
            )

        super().fit(X, convert_dtype=convert_dtype)
        y, classes = preprocess_labels(
            y,
            n_samples=self.n_samples_fit_,
            order="F",
            dtype=np.int32,
            allow_multitarget=True
        )
        self.classes_ = classes
        self._y = y
        return self

    @property
    def outputs_2d_(self):
        """Whether the output is 2d"""
        return self._y.ndim == 2 and self._y.shape[1] != 1

    @generate_docstring(convert_dtype_cast='np.float32',
                        return_values={'name': 'X_new',
                                       'type': 'dense',
                                       'description': 'Labels predicted',
                                       'shape': '(n_samples, 1)'})
    @run_in_internal_context
    def predict(self, X, *, convert_dtype=True):
        """
        Use the trained k-nearest neighbors classifier to
        predict the labels for X

        """
        # Get KNN results - always get distances to compute weights
        knn_distances, knn_indices = self.kneighbors(
            X, return_distance=True, convert_dtype=convert_dtype
        )

        cdef size_t n_rows
        inds, n_rows, _, _ = input_to_cuml_array(
            knn_indices,
            order='C',
            check_dtype=np.int64,
            convert_to_dtype=(np.int64 if convert_dtype else None),
        )

        dists, _, _, _ = input_to_cuml_array(
            knn_distances,
            order='C',
            check_dtype=np.float32,
            convert_to_dtype=(np.float32 if convert_dtype else None),
        )

        # Allocate array for predictions
        out_cols = self._y.shape[1] if self._y.ndim == 2 else 1
        out_shape = (n_rows, out_cols) if out_cols > 1 else n_rows
        out = cp.empty(out_shape, dtype=np.int32, order="C")
        cdef int* out_ptr = <int*><uintptr_t>out.data.ptr

        # Compose vector of y columns
        cdef vector[int*] y_vec
        for i in range(out_cols):
            col = self._y if out_cols == 1 else self._y[:, i]
            y_vec.push_back(<int*><uintptr_t>col.data.ptr)

        # Compute weights (returns None for uniform weights)
        weights_cp = compute_weights(dists.to_output('cupy'), self.weights)
        cdef float* weights_ptr = <float*><uintptr_t>(
            0 if weights_cp is None else weights_cp.data.ptr
        )

        handle = get_handle(model=self)
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
        cdef int64_t* inds_ptr = <int64_t*><uintptr_t>inds.ptr
        cdef size_t n_samples_fit = self._y.shape[0]
        cdef int n_neighbors = self.n_neighbors
        with nogil:
            knn_classify(
                handle_[0],
                out_ptr,
                inds_ptr,
                y_vec,
                n_samples_fit,
                n_rows,
                n_neighbors,
                weights_ptr
            )

        handle.sync()

        with cuml.internals.exit_internal_context():
            output_type = self._get_output_type(X)
        return decode_labels(out, self.classes_, output_type=output_type)

    @generate_docstring(convert_dtype_cast='np.float32',
                        return_values={'name': 'X_new',
                                       'type': 'dense',
                                       'description': 'Labels probabilities',
                                       'shape': '(n_samples, 1)'})
    @reflect
    def predict_proba(self, X, *, convert_dtype=True) -> CumlArray | list[CumlArray]:
        """
        Use the trained k-nearest neighbors classifier to
        predict the label probabilities for X

        """
        # Get KNN results - always get distances to compute weights
        knn_distances, knn_indices = self.kneighbors(
            X, return_distance=True, convert_dtype=convert_dtype
        )

        cdef size_t n_rows
        inds, n_rows, _, _ = input_to_cuml_array(
            knn_indices,
            order='C',
            check_dtype=np.int64,
            convert_to_dtype=(np.int64 if convert_dtype else None)
        )

        dists, _, _, _ = input_to_cuml_array(
            knn_distances,
            order='C',
            check_dtype=np.float32,
            convert_to_dtype=(np.float32 if convert_dtype else None)
        )

        if self._y.ndim == 1 or self._y.shape[1] == 1:
            n_classes = [len(self.classes_)]
            ys = [self._y]
        else:
            n_classes = [len(c) for c in self.classes_]
            ys = [self._y[:, i] for i in range(self._y.shape[1])]

        # Construct vectors of y columns and output probas
        probas = []
        cdef vector[float*] out_vec
        cdef vector[int*] y_vec
        for n, y in zip(n_classes, ys):
            proba = CumlArray.zeros(
                (n_rows, n), dtype=np.float32, order="C", index=inds.index
            )
            probas.append(proba)
            out_vec.push_back(<float*><uintptr_t>proba.ptr)
            y_vec.push_back(<int*><uintptr_t>y.data.ptr)

        # Compute weights (returns None for uniform weights)
        weights_cp = compute_weights(dists.to_output('cupy'), self.weights)
        cdef float* weights_ptr = <float*><uintptr_t>(
            0 if weights_cp is None else weights_cp.data.ptr
        )

        handle = get_handle(model=self)
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
        cdef int64_t* inds_ptr = <int64_t*><uintptr_t>inds.ptr
        cdef size_t n_samples_fit = self._y.shape[0]
        cdef int n_neighbors = self.n_neighbors
        with nogil:
            knn_class_proba(
                handle_[0],
                out_vec,
                inds_ptr,
                y_vec,
                n_samples_fit,
                n_rows,
                n_neighbors,
                weights_ptr
            )
        handle.sync()
        return probas[0] if len(probas) == 1 else probas

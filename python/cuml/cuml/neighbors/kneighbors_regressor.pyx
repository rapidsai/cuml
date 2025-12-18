#
# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np

from cuml.common import input_to_cuml_array
from cuml.common.doc_utils import generate_docstring
from cuml.internals import get_handle, reflect
from cuml.internals.array import CumlArray
from cuml.internals.interop import UnsupportedOnGPU, to_cpu, to_gpu
from cuml.internals.mixins import FMajorInputTagMixin, RegressorMixin
from cuml.neighbors.nearest_neighbors import NearestNeighbors
from cuml.neighbors.weights import compute_weights

from libc.stdint cimport int64_t, uintptr_t
from libcpp.vector cimport vector
from pylibraft.common.handle cimport handle_t


cdef extern from "cuml/neighbors/knn.hpp" namespace "ML" nogil:

    void knn_regress(
        handle_t &handle,
        float *out,
        int64_t *knn_indices,
        vector[float *] &y,
        size_t n_rows,
        size_t n_samples,
        int k,
        float *sample_weight
    ) except +


class KNeighborsRegressor(RegressorMixin, FMajorInputTagMixin, NearestNeighbors):
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
    algorithm : string (default='auto')
        The query algorithm to use. Valid options are:

        - ``'auto'``: to automatically select brute-force or
          random ball cover based on data shape and metric
        - ``'rbc'``: for the random ball algorithm, which partitions
          the data space and uses the triangle inequality to lower the
          number of potential distances. Currently, this algorithm
          supports 2d Euclidean and Haversine.
        - ``'brute'``: for brute-force, slow but produces exact results
        - ``'ivfflat'``: for inverted file, divide the dataset in partitions
          and perform search on relevant partitions only
        - ``'ivfpq'``: for inverted file and product quantization,
          same as inverted list, in addition the vectors are broken
          in n_features/M sub-vectors that will be encoded thanks
          to intermediary k-means clusterings. This encoding provide
          partial information allowing faster distances calculations
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

        >>> from cuml.neighbors import KNeighborsRegressor
        >>> from cuml.datasets import make_regression
        >>> from cuml.model_selection import train_test_split

        >>> X, y = make_regression(n_samples=100, n_features=10,
        ...                        random_state=5)
        >>> X_train, X_test, y_train, y_test = train_test_split(
        ...   X, y, train_size=0.80, random_state=5)

        >>> knn = KNeighborsRegressor(n_neighbors=10)
        >>> knn.fit(X_train, y_train)
        KNeighborsRegressor()
        >>> knn.predict(X_test) # doctest: +SKIP
        array([ 14.770798  ,  51.8834    ,  66.15657   ,  46.978275  ,
            21.589611  , -14.519918  , -60.25534   , -20.856869  ,
            29.869623  , -34.83317   ,   0.45447388, 120.39675   ,
            109.94834   ,  63.57794   , -17.956171  ,  78.77663   ,
            30.412262  ,  32.575233  ,  74.72834   , 122.276855  ],
        dtype=float32)

    Notes
    -----

    For additional docs, see `scikitlearn's KNeighborsClassifier
    <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html>`_.
    """
    _cpu_class_path = "sklearn.neighbors.KNeighborsRegressor"

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
            "_y": to_gpu(model._y, order="F", dtype=np.float32),
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        return {
            "_y": to_cpu(self._y),
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
    def fit(self, X, y, *, convert_dtype=True) -> "KNeighborsRegressor":
        """
        Fit a GPU index for k-nearest neighbors regression model.

        """
        if self.weights not in ('uniform', 'distance', None) and not callable(self.weights):
            raise ValueError(
                f"weights must be 'uniform', 'distance', or a callable, got {self.weights}"
            )
        super().fit(X, convert_dtype=convert_dtype)

        self._y = input_to_cuml_array(
            y,
            order='F',
            check_rows=self.n_samples_fit_,
            check_dtype=np.float32,
            convert_to_dtype=(np.float32 if convert_dtype else None),
        ).array

        return self

    @generate_docstring(convert_dtype_cast='np.float32',
                        return_values={'name': 'X_new',
                                       'type': 'dense',
                                       'description': 'Predicted values',
                                       'shape': '(n_samples, n_features)'})
    @reflect
    def predict(self, X, *, convert_dtype=True) -> CumlArray:
        """
        Use the trained k-nearest neighbors regression model to
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

        dists = input_to_cuml_array(
            knn_distances,
            order='C',
            check_dtype=np.float32,
            convert_to_dtype=(np.float32 if convert_dtype else None),
        ).array

        cdef int64_t* inds_ctype = <int64_t*><uintptr_t>inds.ptr

        res_cols = 1 if self._y.ndim == 1 else self._y.shape[1]
        res_shape = n_rows if res_cols == 1 else (n_rows, res_cols)

        out = CumlArray.zeros(
            res_shape, dtype=np.float32, order="C", index=inds.index
        )

        cdef float* out_ptr = <float*><uintptr_t>out.ptr

        cdef vector[float*] y_vec
        cdef float* y_ptr
        for col_num in range(res_cols):
            col = self._y if res_cols == 1 else self._y[:, col_num]
            y_ptr = <float*><uintptr_t>col.ptr
            y_vec.push_back(y_ptr)

        handle = get_handle(model=self)
        cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()

        # Compute weights (returns None for uniform weights)
        weights_cp = compute_weights(dists.to_output('cupy'), self.weights)
        cdef float* weights_ctype = <float*><uintptr_t>(
            0 if weights_cp is None else weights_cp.data.ptr
        )

        cdef size_t n_samples_fit = self._y.shape[0]
        cdef int n_neighbors = self.n_neighbors
        with nogil:
            knn_regress(
                handle_[0],
                out_ptr,
                inds_ctype,
                y_vec,
                n_samples_fit,
                n_rows,
                n_neighbors,
                weights_ctype
            )

        handle.sync()

        return out

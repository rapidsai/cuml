#
# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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
import enum
import warnings

import cupy as cp
import numpy as np
from pylibraft.common.handle import Handle

from cuml.common import input_to_cuml_array
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.doc_utils import generate_docstring
from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from cuml.internals.interop import (
    InteropMixin,
    UnsupportedOnGPU,
    to_cpu,
    to_gpu,
)
from cuml.internals.mixins import FMajorInputTagMixin, RegressorMixin
from cuml.linear_model.base import LinearPredictMixin

from libc.stdint cimport uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t


cdef extern from "cuml/linear_model/glm.hpp" namespace "ML::GLM" nogil:

    cdef void olsFit(handle_t& handle,
                     float *input,
                     size_t n_rows,
                     size_t n_cols,
                     float *labels,
                     float *coef,
                     float *intercept,
                     bool fit_intercept,
                     bool normalize,
                     int algo,
                     float *sample_weight) except +

    cdef void olsFit(handle_t& handle,
                     double *input,
                     size_t n_rows,
                     size_t n_cols,
                     double *labels,
                     double *coef,
                     double *intercept,
                     bool fit_intercept,
                     bool normalize,
                     int algo,
                     double *sample_weight) except +


class Algo(enum.IntEnum):
    """The lstsq solver algorithm"""
    SVD = 0
    EIG = 1
    QR = 2
    SVD_QR = 3

    @classmethod
    def parse(cls, name):
        out = {
            "svd": cls.SVD,
            "eig": cls.EIG,
            "qr": cls.QR,
            "svd-qr": cls.SVD_QR,
            "svd-jacobi": cls.SVD
        }.get(name)
        if out is None:
            raise ValueError("algorithm {name!r} is not supported")
        return out


# 1e-10 chosen to match C++ implementation
_divide_non_zero = cp.ElementwiseKernel(
    "T x, T y",
    "T z",
    "z = abs(y) < 1e-10 ? x : x / y",
    "divide_non_zero"
)


def _cuda_ptr(X):
    """Returns a pointer to a backing device array, or None if not a device array"""
    if (interface := getattr(X, "__cuda_array_interface__", None)) is not None:
        return interface["data"][0]
    return None


class LinearRegression(Base,
                       InteropMixin,
                       LinearPredictMixin,
                       RegressorMixin,
                       FMajorInputTagMixin):
    """
    LinearRegression is a simple machine learning model where the response y is
    modelled by a linear combination of the predictors in X.

    Examples
    --------

    .. code-block:: python

        >>> import cupy as cp
        >>> import cudf

        >>> # Both import methods supported
        >>> from cuml import LinearRegression
        >>> from cuml.linear_model import LinearRegression
        >>> lr = LinearRegression(fit_intercept = True, normalize = False,
        ...                       algorithm = "eig")
        >>> X = cudf.DataFrame()
        >>> X['col1'] = cp.array([1,1,2,2], dtype=cp.float32)
        >>> X['col2'] = cp.array([1,2,2,3], dtype=cp.float32)
        >>> y = cudf.Series(cp.array([6.0, 8.0, 9.0, 11.0], dtype=cp.float32))
        >>> reg = lr.fit(X,y)
        >>> print(reg.coef_)
        0   1.0
        1   2.0
        dtype: float32
        >>> print(reg.intercept_)
        3.0...

        >>> X_new = cudf.DataFrame()
        >>> X_new['col1'] = cp.array([3,2], dtype=cp.float32)
        >>> X_new['col2'] = cp.array([5,5], dtype=cp.float32)
        >>> preds = lr.predict(X_new)
        >>> print(preds) # doctest: +SKIP
        0   15.999...
        1   14.999...
        dtype: float32


    Parameters
    ----------
    algorithm : {'auto', 'svd', 'eig', 'qr', 'svd-qr', 'svd-jacobi'}, (default = 'auto')
        Choose an algorithm:

          * 'auto' - 'eig', or 'svd' if y multi-target or X has only one column
          * 'svd' - alias for svd-jacobi
          * 'eig' - use an eigendecomposition of the covariance matrix
          * 'qr'  - use QR decomposition algorithm and solve `Rx = Q^T y`
          * 'svd-qr' - compute SVD decomposition using QR algorithm
          * 'svd-jacobi' - compute SVD decomposition using Jacobi iterations

        Among these algorithms, only 'svd-jacobi' supports the case when the
        number of features is larger than the sample size; this algorithm
        is force-selected automatically in such a case.

        For the broad range of inputs, 'eig' and 'qr' are usually the fastest,
        followed by 'svd-jacobi' and then 'svd-qr'. In theory, SVD-based
        algorithms are more stable.
    fit_intercept : boolean (default = True)
        If True, LinearRegression tries to correct for the global mean of y.
        If False, the model expects that you have centered the data.
    copy_X : bool, default=True
        If True, cuml will copy X when needed to avoid mutating the input array.
        If you're ok with X being overwritten, setting to False may avoid a copy,
        reducing memory usage for certain algorithms.
    normalize : boolean (default = False)
        This parameter is ignored when `fit_intercept` is set to False.
        If True, the predictors in X will be normalized by dividing by the
        column-wise standard deviation.
        If False, no scaling will be done.
        Note: this is in contrast to sklearn's deprecated `normalize` flag,
        which divides by the column-wise L2 norm; but this is the same as if
        using sklearn's StandardScaler.
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

    Attributes
    ----------
    coef_ : array, shape (n_features)
        The estimated coefficients for the linear regression model.
    intercept_ : array
        The independent term. If `fit_intercept` is False, will be 0.

    Notes
    -----
    LinearRegression suffers from multicollinearity (when columns are
    correlated with each other), and variance explosions from outliers.
    Consider using Ridge Regression to fix the multicollinearity problem, and
    consider maybe first DBSCAN to remove the outliers, or statistical analysis
    to filter possible outliers.

    **Applications of LinearRegression**

        LinearRegression is used in regression tasks where one wants to predict
        say sales or house prices. It is also used in extrapolation or time
        series tasks, dynamic systems modelling and many other machine learning
        tasks. This model should be first tried if the machine learning problem
        is a regression task (predicting a continuous variable).

    For additional information, see `scikitlearn's OLS documentation
    <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`__.

    For an additional example see `the OLS notebook
    <https://github.com/rapidsai/cuml/blob/main/notebooks/linear_regression_demo.ipynb>`__.
    """

    coef_ = CumlArrayDescriptor(order="F")
    intercept_ = CumlArrayDescriptor(order="F")

    _cpu_class_path = "sklearn.linear_model.LinearRegression"

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "algorithm",
            "fit_intercept",
            "copy_X",
            "normalize",
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        if model.positive:
            raise UnsupportedOnGPU("`positive=True` is not supported")

        return {
            "fit_intercept": model.fit_intercept,
            "copy_X": model.copy_X,
        }

    def _params_to_cpu(self):
        return {
            "fit_intercept": self.fit_intercept,
            "copy_X": self.copy_X,
        }

    def _attrs_from_cpu(self, model):
        return {
            "intercept_": to_gpu(model.intercept_, order="F"),
            "coef_": to_gpu(model.coef_, order="F"),
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        return {
            "intercept_": to_cpu(self.intercept_),
            "coef_": to_cpu(self.coef_),
            **super()._attrs_to_cpu(model),
        }

    def __init__(
        self,
        *,
        algorithm="auto",
        fit_intercept=True,
        copy_X=True,
        normalize=False,
        handle=None,
        verbose=False,
        output_type=None
    ):
        if handle is None and algorithm in ("auto", "eig"):
            # if possible, create two streams, so that eigenvalue decomposition
            # can benefit from running independent operations concurrently.
            handle = Handle(n_streams=2)

        super().__init__(handle=handle, verbose=verbose, output_type=output_type)

        self.algorithm = algorithm
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.normalize = normalize

    def _select_algo(self, X, y):
        """Select the solver algorithm based on `algorithm` and problem dimensions"""
        if X.shape[0] == 1:
            fallback_reason = "single-column X"
        elif y.ndim == 2 and y.shape[1] > 1:
            fallback_reason = "multi-column y"
        else:
            fallback_reason = None

        if self.algorithm == "auto":
            algo = Algo.SVD if fallback_reason else Algo.EIG
        else:
            algo = Algo.parse(self.algorithm)
            if fallback_reason and algo != Algo.SVD:
                warnings.warn(
                    (
                        "Falling back to `algorithm='svd'` as `algorithm="
                        "{self.algorithm!r}` doesn't support {fallback_reason}."
                    ),
                    UserWarning,
                )
                algo = Algo.SVD
        return algo

    @generate_docstring()
    def fit(self, X, y, sample_weight=None, *, convert_dtype=True) -> "LinearRegression":
        """
        Fit the model with X and y.

        """
        X_m = input_to_cuml_array(
            X,
            convert_to_dtype=(np.float32 if convert_dtype else None),
            check_dtype=[np.float32, np.float64],
            order="F",
        ).array

        if X_m.shape[0] < 2:
            raise ValueError("X matrix must have at least two rows")

        if X_m.shape[1] < 1:
            raise ValueError("X matrix must have at least one column")

        y_m = input_to_cuml_array(
            y,
            check_dtype=X_m.dtype,
            convert_to_dtype=(X_m.dtype if convert_dtype else None),
            check_rows=X_m.shape[0],
            order="F",
        ).array

        if sample_weight is not None:
            # Always copy the weights, all solvers mutate them
            sample_weight = input_to_cuml_array(
                sample_weight,
                check_dtype=X_m.dtype,
                convert_to_dtype=(X_m.dtype if convert_dtype else None),
                check_rows=X_m.shape[0],
                check_cols=1,
                order="F",
                deepcopy=True,
            ).array

        cdef int algo = self._select_algo(X_m, y_m)

        X_is_copy = _cuda_ptr(X) != X_m.ptr
        y_is_copy = _cuda_ptr(y) != y_m.ptr

        if y_m.ndim > 1 and y_m.shape[1] > 1:
            # Fallback to cupy SVD implementation for multi-target problems
            self._fit_multi_target(
                X_m, y_m, sample_weight, X_is_copy=X_is_copy, y_is_copy=y_is_copy
            )
            return self

        # All libcuml solvers mutate the inputs. Here we make a copy requested
        # (and one wasn't already made).
        if not X_is_copy and self.copy_X:
            X_m = input_to_cuml_array(X_m, deepcopy=True).array
        if not y_is_copy:
            y_m = input_to_cuml_array(y_m, deepcopy=True).array

        coef = CumlArray.zeros(X_m.shape[1], dtype=X_m.dtype)

        cdef size_t n_rows = X_m.shape[0]
        cdef size_t n_cols = X_m.shape[1]
        cdef uintptr_t X_ptr = X_m.ptr
        cdef uintptr_t y_ptr = y_m.ptr
        cdef uintptr_t sample_weight_ptr = (
            0 if sample_weight is None else sample_weight.ptr
        )
        cdef uintptr_t coef_ptr = coef.ptr
        cdef bool is_float32 = X_m.dtype == np.float32
        cdef float intercept_f32
        cdef double intercept_f64
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        cdef bool fit_intercept = self.fit_intercept
        cdef bool normalize = self.normalize

        with nogil:
            if is_float32:
                olsFit(
                    handle_[0],
                    <float*>X_ptr,
                    n_rows,
                    n_cols,
                    <float*>y_ptr,
                    <float*>coef_ptr,
                    &intercept_f32,
                    fit_intercept,
                    normalize,
                    algo,
                    <float*>sample_weight_ptr,
                )
            else:
                olsFit(
                    handle_[0],
                    <double*>X_ptr,
                    n_rows,
                    n_cols,
                    <double*>y_ptr,
                    <double*>coef_ptr,
                    &intercept_f64,
                    fit_intercept,
                    normalize,
                    algo,
                    <double*>sample_weight_ptr,
                )
        self.handle.sync()

        self.intercept_ = intercept_f32 if is_float32 else intercept_f64
        self.coef_ = coef

        return self

    def _fit_multi_target(
        self, X_m, y_m, sample_weight_m=None, X_is_copy=False, y_is_copy=False,
    ):
        if self.normalize:
            raise ValueError(
                "The normalize option is not supported when `y` has "
                "multiple columns."
            )

        X = X_m.to_output("cupy")
        y = y_m.to_output("cupy")

        if self.fit_intercept:
            # Add column containing ones to fit intercept.
            nrow, ncol = X.shape
            X_temp = cp.empty_like(X, shape=(nrow, ncol + 1))
            X_temp[:, :ncol] = X
            X_temp[:, ncol] = 1.
            X = X_temp
            X_is_copy = True

        if sample_weight_m is not None:
            sample_weight = sample_weight_m.to_output("cupy")
            # Weights are always copied, can mutate buffer
            weight_sqrt = cp.sqrt(sample_weight, out=sample_weight)
            # Multiply by weights, reusing existing buffers when possible
            X = cp.multiply(
                X,
                weight_sqrt[:, None],
                out=X if X_is_copy or not self.copy_X else None,
            )
            y = cp.multiply(
                y,
                weight_sqrt[:, None],
                out=y if y_is_copy else None
            )

        u, s, vh = cp.linalg.svd(X, full_matrices=False)
        temp = _divide_non_zero(u.T.dot(y), s[:, None])
        coef = vh.T.dot(temp)

        if self.fit_intercept:
            intercept = CumlArray(data=coef[-1])
            coef = CumlArray(data=coef[:-1].T)
        else:
            intercept = 0.0
            coef = CumlArray(data=coef.T)

        self.coef_ = coef
        self.intercept_ = intercept

    @staticmethod
    def _more_static_tags():
        return {"multioutput": True}

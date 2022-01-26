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

import numpy as np
import warnings
from cupy import linalg
import cupy as cp
from cupyx import lapack, geterr, seterr
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.base import Base
from cuml.common.mixins import RegressorMixin
from cuml.common.doc_utils import generate_docstring
from cuml.common import input_to_cuml_array

from cuml.metrics import pairwise_kernels



# cholesky solve with fallback to least squares for singular problems
def _safe_solve(K, y):
    try:
        # we need to set the error mode of cupy to raise
        # otherwise we silently get an array of NaNs
        err_mode = geterr()["linalg"]
        seterr(linalg="raise")
        dual_coef = lapack.posv(K, y)
        seterr(linalg=err_mode)
    except np.linalg.LinAlgError:
        warnings.warn(
            "Singular matrix in solving dual problem. Using "
            "least-squares solution instead."
        )
        dual_coef = linalg.lstsq(K, y, rcond=None)[0]
    return dual_coef


def _solve_cholesky_kernel(K, y, alpha, sample_weight=None):
    # dual_coef = inv(X X^t + alpha*Id) y
    n_samples = K.shape[0]
    n_targets = y.shape[1]

    K = cp.array(K, dtype=np.float64)

    alpha = cp.atleast_1d(alpha)
    one_alpha = alpha.size == 1
    has_sw = sample_weight is not None

    if has_sw:
        # Unlike other solvers, we need to support sample_weight directly
        # because K might be a pre-computed kernel.
        sw = cp.sqrt(cp.atleast_1d(sample_weight))
        y = y * sw[:, cp.newaxis]
        K *= cp.outer(sw, sw)

    if one_alpha:
        # Only one penalty, we can solve multi-target problems in one time.
        K.flat[:: n_samples + 1] += alpha[0]

        dual_coef = _safe_solve(K, y)

        if has_sw:
            dual_coef *= sw[:, cp.newaxis]

        return dual_coef
    else:
        # One penalty per target. We need to solve each target separately.
        dual_coefs = cp.empty([n_targets, n_samples], K.dtype)

        for dual_coef, target, current_alpha in zip(dual_coefs, y.T, alpha):
            K.flat[:: n_samples + 1] += current_alpha

            dual_coef[:] = _safe_solve(K, target).ravel()

            K.flat[:: n_samples + 1] -= current_alpha

        if has_sw:
            dual_coefs *= sw[cp.newaxis, :]

        return dual_coefs.T




class KernelRidge(Base, RegressorMixin):
    dual_coef_ = CumlArrayDescriptor()

    def __init__(
        self,
        *,
        alpha=1,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
        kernel_params=None,
        handle=None,
        output_type=None,
        verbose=False
    ):
        super().__init__(handle=handle, verbose=verbose, output_type=output_type)
        self.alpha = cp.asarray(alpha)
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params

    def _get_kernel(self, X, Y=None):
        if isinstance(self.kernel, str):
            params = {"gamma": self.gamma, "degree": self.degree, "coef0": self.coef0}
        else:
            params = self.kernel_params or {}
        return pairwise_kernels(X, metric=self.kernel, filter_params=True, **params)

    @generate_docstring()
    def fit(self, X, y, sample_weight=None, convert_dtype=True) -> "KernelRidge":
        """
        Fit the model with X and y.

        """

        ravel = False
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            ravel = True

        X_m, n_rows, self.n_cols, self.dtype = input_to_cuml_array(
            X, check_dtype=[np.float32, np.float64]
        )

        y_m, _, _, _ = input_to_cuml_array(
            y,
            check_dtype=self.dtype,
            convert_to_dtype=(self.dtype if convert_dtype else None),
            check_rows=n_rows,
        )

        if self.n_cols < 1:
            msg = "X matrix must have at least a column"
            raise TypeError(msg)

        K = self._get_kernel(X_m, self.kernel)
        self.dual_coef_ = _solve_cholesky_kernel(
            K, cp.asarray(y_m), self.alpha, sample_weight
        )

        if ravel:
            self.dual_coef_ = self.dual_coef_.ravel()
        self.X_fit_ = X_m
        return self

    def predict(self, X):
        """Predict using the kernel ridge model.
            Parameters
            ----------
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                Samples. If kernel == "precomputed" this is instead a
                precomputed kernel matrix, shape = [n_samples,
                n_samples_fitted], where n_samples_fitted is the number of
                samples used in the fitting for this estimator.
            Returns
            -------
            C : array of shape (n_samples,) or (n_samples, n_targets)
                Returns predicted values.
            """
        X_m, _, _, _ = input_to_cuml_array(X, check_dtype=[np.float32, np.float64])

        K = self._get_kernel(X_m, self.X_fit_)
        return cp.dot(cp.asarray(K), cp.asarray(self.dual_coef_))


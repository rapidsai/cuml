# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import cupy as cp
import cupyx.scipy.sparse

import cuml.internals
from cuml.common.doc_utils import generate_docstring
from cuml.common.sparse_utils import is_sparse
from cuml.internals.array import CumlArray
from cuml.internals.array_sparse import SparseCumlArray
from cuml.internals.input_utils import input_to_cuml_array
from cuml.internals.validation import check_features, check_is_fitted


class LinearPredictMixin:
    @generate_docstring(
        return_values={
            "name": "preds",
            "type": "dense",
            "description": "Predicted values",
            "shape": "(n_samples, 1)",
        }
    )
    @cuml.internals.reflect
    def predict(self, X, *, convert_dtype=True) -> CumlArray:
        """
        Predicts `y` values for `X`.
        """
        check_is_fitted(self)
        check_features(self, X)

        if is_sparse(X):
            X_m = SparseCumlArray(X, convert_to_dtype=self.coef_.dtype)
        else:
            X_m = input_to_cuml_array(
                X,
                check_dtype=self.coef_.dtype,
                convert_to_dtype=(self.coef_.dtype if convert_dtype else None),
                check_cols=self.n_features_in_,
                order="K",
            ).array

        X = X_m.to_output("cupy")
        coef = self.coef_.to_output("cupy")
        intercept = self.intercept_
        if isinstance(intercept, CumlArray):
            intercept = intercept.to_output("cupy")

        out = X @ coef.T
        out += intercept

        return CumlArray(out, index=X_m.index)


class LinearClassifierMixin:
    @generate_docstring(
        X="dense_sparse",
        return_values={
            "name": "scores",
            "type": "dense",
            "description": "Confidence scores",
            "shape": "(n_samples,) or (n_samples, n_classes)",
        },
    )
    @cuml.internals.reflect
    def decision_function(self, X, *, convert_dtype=True) -> CumlArray:
        """Predict confidence scores for samples."""
        check_is_fitted(self)
        check_features(self, X)

        if is_sparse(X):
            X_m = SparseCumlArray(X, convert_to_dtype=self.coef_.dtype)
        else:
            X_m = input_to_cuml_array(
                X,
                check_dtype=self.coef_.dtype,
                convert_to_dtype=(self.coef_.dtype if convert_dtype else None),
                check_cols=self.n_features_in_,
                order="K",
            ).array

        X = X_m.to_output("cupy")
        coef = self.coef_.to_output("cupy")
        intercept = self.intercept_
        if isinstance(intercept, CumlArray):
            intercept = intercept.to_output("cupy")

        out = X @ coef.T
        out += intercept

        if out.ndim > 1 and out.shape[1] == 1:
            out = out.reshape(-1)

        return CumlArray(out, index=X_m.index)


def center_and_scale(
    X,
    y,
    sample_weight=None,
    fit_intercept=True,
    may_mutate_X=False,
    may_mutate_y=False,
):
    """Common preprocessing for X and y for fitting a linear model.

    Performs centering and scaling of X and y.

    Parameters
    ----------
    X : dense or sparse array, shape (n_samples, n_features)
        The features.
    y : dense array, shape (n_samples,) or (n_samples, n_targets)
        The targets.
    sample_weight : cp.ndarray or None
        The sample weights.
    fit_intercept : bool
        Whether to fit an intercept.
    may_mutate_X : bool
        Whether to allow mutating X inplace to save memory when possible.
    may_mutate_y : bool
        Whether to allow mutating y inplace to save memory when possible.

    Returns
    -------
    X : cupy.ndarray or cupyx.scipy.sparse.sp_matrix, shape (n_samples, n_features)
        Rescaled by sample weights. Also centered if dense.
    y : cupy.ndarray, shape (n_samples, 1) or (n_samples, n_targets)
        Rescaled by sample weights and centered.
    X_offset : cupy.ndarray or None, shape (n_features,)
        The per-column mean of X, or None if ``fit_intercept=False``.
    y_offset : cupy.ndarray or None, shape (n_targets,)
        The per-column mean of y, or None if ``fit_intercept=False``.
    sample_weight_sqrt : cupy.ndarray or None, shape (n_samples,)
        The sqrt of the ``sample_weight``, or None if unweighted.
    """
    X_is_sparse = cupyx.scipy.sparse.issparse(X)

    # Ensure 2D
    if X.ndim == 1:
        X = X[:, None]
    if y.ndim == 1:
        y = y[:, None]

    if fit_intercept:
        if sample_weight is not None:
            # Offset by weighted mean
            den = sample_weight.sum()
            if X_is_sparse:
                X_offset = (
                    X.multiply(sample_weight[:, None]).sum(axis=0).ravel()
                    / den
                )
            else:
                X_offset = (X * sample_weight[:, None]).sum(axis=0) / den
            y_offset = (y * sample_weight[:, None]).sum(axis=0) / den
        else:
            # Offset by mean
            X_offset = X.mean(axis=0).ravel()
            y_offset = y.mean(axis=0)

        # Subtract offset, reusing existing buffers when possible
        if not X_is_sparse:
            # Don't offset sparse X since that would remove sparsity.
            # Instead that's handled later in the solvers.
            X = cp.subtract(
                X,
                X_offset,
                out=X if may_mutate_X else None,
            )
            may_mutate_X = True
        y = cp.subtract(y, y_offset, out=y if may_mutate_y else None)
        may_mutate_y = True
    else:
        X_offset = y_offset = None

    if sample_weight is not None:
        sqrt_weight = cp.sqrt(sample_weight)
        # Multiply by sqrt(weight), reusing existing buffers when possible
        if X_is_sparse:
            X = X.multiply(sqrt_weight[:, None])
        else:
            X = cp.multiply(
                X,
                sqrt_weight[:, None],
                out=X if may_mutate_X else None,
            )
        y = cp.multiply(
            y, sqrt_weight[:, None], out=y if may_mutate_y else None
        )
    else:
        sqrt_weight = None

    return X, y, X_offset, y_offset, sqrt_weight

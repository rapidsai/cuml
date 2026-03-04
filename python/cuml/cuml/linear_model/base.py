#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import cuml.internals
from cuml.common.doc_utils import generate_docstring
from cuml.internals.array import CumlArray
from cuml.internals.validation import check_is_fitted, check_X


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
        X = check_X(
            self,
            X,
            dtype=self.coef_.dtype,
            convert_dtype=convert_dtype,
        )
        X_cp = X.to_output("cupy")

        coef = self.coef_.to_output("cupy")

        intercept = self.intercept_
        if isinstance(intercept, CumlArray):
            intercept = intercept.to_output("cupy")

        out = X_cp @ coef.T
        out += intercept

        return CumlArray(out, index=X.index)


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
        X_m = check_X(
            self,
            X,
            dtype=self.coef_.dtype,
            convert_dtype=convert_dtype,
        )
        X = X_m.to_output("cupy")
        out_index = X_m.index if isinstance(X_m, CumlArray) else None

        coef = self.coef_.to_output("cupy")
        intercept = self.intercept_
        if isinstance(intercept, CumlArray):
            intercept = intercept.to_output("cupy")

        out = X @ coef.T
        out += intercept

        if out.ndim > 1 and out.shape[1] == 1:
            out = out.reshape(-1)

        return CumlArray(out, index=out_index)

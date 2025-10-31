#
# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import warnings

import cuml.internals
from cuml.common.doc_utils import generate_docstring
from cuml.internals.array import CumlArray
from cuml.internals.input_utils import input_to_cuml_array


class LinearPredictMixin:
    @generate_docstring(
        return_values={
            "name": "preds",
            "type": "dense",
            "description": "Predicted values",
            "shape": "(n_samples, 1)",
        }
    )
    @cuml.internals.api_base_return_array_skipall
    def predict(self, X, *, convert_dtype=True) -> CumlArray:
        """
        Predicts `y` values for `X`.
        """
        if getattr(self, "coef_", None) is None:
            raise ValueError(
                "LinearModel.predict() cannot be called before fit(). "
                "Please fit the model first."
            )

        X = input_to_cuml_array(
            X,
            check_dtype=self.coef_.dtype,
            convert_to_dtype=(self.coef_.dtype if convert_dtype else None),
            check_cols=self.n_features_in_,
            order="K",
        ).array
        X_cp = X.to_output("cupy")

        coef = self.coef_.to_output("cupy")

        intercept = self.intercept_
        if isinstance(intercept, CumlArray):
            intercept = intercept.to_output("cupy")

        out = X_cp @ coef.T
        out += intercept

        return CumlArray(out, index=X.index)


def check_deprecated_normalize(model):
    """Warn if the deprecated `normalize` option is used."""
    if model.normalize:
        cls_name = type(model).__name__
        warnings.warn(
            (
                f"The `normalize` option to `{cls_name}` was deprecated in "
                f"25.12 and will be removed in 26.02. Please use a `StandardScaler` "
                f"to normalize your data external to `{cls_name}`."
            ),
            FutureWarning,
        )

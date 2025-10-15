#
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

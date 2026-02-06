#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""Thin Pipeline wrapper for cuml.accel that keeps intermediate data on device."""

from __future__ import annotations

from typing import Any, Callable

from sklearn.pipeline import Pipeline as _SklearnPipeline

from cuml.internals.global_settings import GlobalSettings
from cuml.internals.interop import InteropMixin
from cuml.internals.outputs import coerce_arrays, using_output_type

__all__ = ("Pipeline",)


class Pipeline(_SklearnPipeline, InteropMixin):
    """Pipeline that keeps intermediate data on device during fit/predict/transform.

    Subclasses sklearn.pipeline.Pipeline and InteropMixin. Runs all chained
    steps under using_output_type("cupy") so data stays on GPU between steps,
    then converts final outputs back to the user's expected type.
    """

    _cpu_class_path = "sklearn.pipeline.Pipeline"

    @classmethod
    def _params_from_cpu(cls, model) -> dict[str, Any]:
        return model.get_params(deep=False)

    def _params_to_cpu(self) -> dict[str, Any]:
        return self.get_params(deep=False)

    def _attrs_from_cpu(self, model) -> dict[str, Any]:
        return super()._attrs_from_cpu(model)

    def _attrs_to_cpu(self, model) -> dict[str, Any]:
        return super()._attrs_to_cpu(model)

    def _run_with_cupy_output(
        self,
        super_method: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Run a parent method with cupy output type, then convert result if needed."""
        output_type = GlobalSettings().output_type or "numpy"
        with using_output_type("cupy"):
            result = super_method(*args, **kwargs)
        if result is self:
            return self
        return coerce_arrays(result, output_type)

    def fit(self, X, y=None, **params):
        with using_output_type("cupy"):
            return super().fit(X, y, **params)

    def fit_transform(self, X, y=None, **params):
        return self._run_with_cupy_output(
            super().fit_transform, X, y, **params
        )

    def fit_predict(self, X, y=None, **params):
        return self._run_with_cupy_output(super().fit_predict, X, y, **params)

    def predict(self, X, **params):
        return self._run_with_cupy_output(super().predict, X, **params)

    def predict_proba(self, X, **params):
        return self._run_with_cupy_output(super().predict_proba, X, **params)

    def predict_log_proba(self, X, **params):
        return self._run_with_cupy_output(
            super().predict_log_proba, X, **params
        )

    def decision_function(self, X, **params):
        return self._run_with_cupy_output(
            super().decision_function, X, **params
        )

    def score_samples(self, X):
        return self._run_with_cupy_output(super().score_samples, X)

    def transform(self, X, **params):
        return self._run_with_cupy_output(super().transform, X, **params)

    def inverse_transform(self, X, **params):
        return self._run_with_cupy_output(
            super().inverse_transform, X, **params
        )

    def score(self, X, y=None, sample_weight=None, **params):
        return self._run_with_cupy_output(
            super().score, X, y, sample_weight=sample_weight, **params
        )

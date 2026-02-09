#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""Thin Pipeline wrapper for cuml.accel that keeps intermediate data on device."""

from __future__ import annotations

from typing import Any

from sklearn.base import clone
from sklearn.pipeline import Pipeline as _SklearnPipeline
from sklearn.pipeline import _fit_transform_one
from sklearn.utils._user_interface import _print_elapsed_time
from sklearn.utils.metadata_routing import (
    _raise_for_params,
    _routing_enabled,
    process_routing,
)
from sklearn.utils.validation import check_is_fitted, check_memory

from cuml.accel import is_proxy
from cuml.internals.global_settings import GlobalSettings
from cuml.internals.interop import InteropMixin
from cuml.internals.outputs import coerce_arrays, using_output_type

__all__ = ("Pipeline",)


class Pipeline(_SklearnPipeline, InteropMixin):
    """Pipeline that keeps intermediate data on device during fit/predict/transform.

    Subclasses sklearn.pipeline.Pipeline and InteropMixin. Intermediate steps
    output cupy arrays only when the next step is a cuml.accel-wrapped
    estimator; otherwise output is numpy so non-accelerated steps receive
    host arrays. Final outputs are coerced to the user's expected type.
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

    @staticmethod
    def _is_accelerated(estimator) -> bool:
        """True if the estimator is a cuml.accel-wrapped (ProxyBase) instance."""
        return is_proxy(estimator)

    def _output_type_for_step(self, step_idx: int):
        """Context manager: cupy if next non-passthrough step is accelerated, else numpy."""
        next_step = None
        for i in range(step_idx + 1, len(self.steps)):
            _, trans = self.steps[i]
            if trans is not None and trans != "passthrough":
                next_step = trans
                break
        output_type = (
            "cupy"
            if (next_step is not None and self._is_accelerated(next_step))
            else "numpy"
        )
        return using_output_type(output_type)

    def _output_type_for_inverse_step(self, step_idx: int):
        """Context manager for inverse_transform: cupy if previous step is accelerated."""
        prev_step = self.steps[step_idx - 1][1] if step_idx > 0 else None
        output_type = (
            "cupy"
            if (prev_step is not None and self._is_accelerated(prev_step))
            else "numpy"
        )
        return using_output_type(output_type)

    def _fit(self, X, y=None, routed_params=None, raw_params=None):
        """Fit the pipeline except the last step, with per-step output type control."""
        if routed_params is None:
            routed_params = {}
        self.steps = list(self.steps)
        self._validate_steps()
        memory = check_memory(self.memory)
        fit_transform_one_cached = memory.cache(_fit_transform_one)

        for step_idx, name, transformer in self._iter(
            with_final=False, filter_passthrough=False
        ):
            if transformer is None or transformer == "passthrough":
                with _print_elapsed_time(
                    "Pipeline", self._log_message(step_idx)
                ):
                    continue

            if hasattr(memory, "location") and memory.location is None:
                cloned_transformer = transformer
            else:
                cloned_transformer = clone(transformer)
            step_params = self._get_metadata_for_step(
                step_idx=step_idx,
                step_params=routed_params.get(name, {}),
                all_params=raw_params,
            )

            with self._output_type_for_step(step_idx):
                X, fitted_transformer = fit_transform_one_cached(
                    cloned_transformer,
                    X,
                    y,
                    weight=None,
                    message_clsname="Pipeline",
                    message=self._log_message(step_idx),
                    params=step_params,
                )
            self.steps[step_idx] = (name, fitted_transformer)
        return X

    def _transform_steps(self, X, routed_params=None):
        """Transform X through intermediate steps only, with per-step output type."""
        Xt = X
        for step_idx, name, transform in self._iter(with_final=False):
            with self._output_type_for_step(step_idx):
                if routed_params is not None:
                    Xt = transform.transform(
                        Xt, **routed_params[name].transform
                    )
                else:
                    Xt = transform.transform(Xt)
        return Xt

    def fit(self, X, y=None, **params):
        return super().fit(X, y, **params)

    def fit_transform(self, X, y=None, **params):
        output_type = GlobalSettings().output_type or "numpy"
        routed_params = self._check_method_params(
            method="fit_transform", props=params
        )
        Xt = self._fit(X, y, routed_params, raw_params=params)
        last_step = self._final_estimator
        with _print_elapsed_time(
            "Pipeline", self._log_message(len(self.steps) - 1)
        ):
            if last_step == "passthrough":
                return coerce_arrays(Xt, output_type)
            last_step_params = self._get_metadata_for_step(
                step_idx=len(self) - 1,
                step_params=routed_params[self.steps[-1][0]],
                all_params=params,
            )
            with using_output_type("cupy"):
                if hasattr(last_step, "fit_transform"):
                    result = last_step.fit_transform(
                        Xt, y, **last_step_params["fit_transform"]
                    )
                else:
                    result = last_step.fit(
                        Xt, y, **last_step_params["fit"]
                    ).transform(Xt, **last_step_params["transform"])
        return coerce_arrays(result, output_type)

    def fit_predict(self, X, y=None, **params):
        self.fit(X, y, **params)
        return self.predict(X, **params)

    def predict(self, X, **params):
        output_type = GlobalSettings().output_type or "numpy"
        check_is_fitted(self)
        if not _routing_enabled():
            Xt = self._transform_steps(X)
            with using_output_type("cupy"):
                result = self.steps[-1][1].predict(Xt, **params)
        else:
            routed_params = process_routing(self, "predict", **params)
            Xt = self._transform_steps(X, routed_params)
            with using_output_type("cupy"):
                result = self.steps[-1][1].predict(
                    Xt, **routed_params[self.steps[-1][0]].predict
                )
        return coerce_arrays(result, output_type)

    def predict_proba(self, X, **params):
        output_type = GlobalSettings().output_type or "numpy"
        check_is_fitted(self)
        if not _routing_enabled():
            Xt = self._transform_steps(X)
            with using_output_type("cupy"):
                result = self.steps[-1][1].predict_proba(Xt, **params)
        else:
            routed_params = process_routing(self, "predict_proba", **params)
            Xt = self._transform_steps(X, routed_params)
            with using_output_type("cupy"):
                result = self.steps[-1][1].predict_proba(
                    Xt, **routed_params[self.steps[-1][0]].predict_proba
                )
        return coerce_arrays(result, output_type)

    def predict_log_proba(self, X, **params):
        output_type = GlobalSettings().output_type or "numpy"
        check_is_fitted(self)
        if not _routing_enabled():
            Xt = self._transform_steps(X)
            with using_output_type("cupy"):
                result = self.steps[-1][1].predict_log_proba(Xt, **params)
        else:
            routed_params = process_routing(
                self, "predict_log_proba", **params
            )
            Xt = self._transform_steps(X, routed_params)
            with using_output_type("cupy"):
                result = self.steps[-1][1].predict_log_proba(
                    Xt, **routed_params[self.steps[-1][0]].predict_log_proba
                )
        return coerce_arrays(result, output_type)

    def decision_function(self, X, **params):
        output_type = GlobalSettings().output_type or "numpy"
        check_is_fitted(self)
        _raise_for_params(params, self, "decision_function")
        routed_params = process_routing(self, "decision_function", **params)
        Xt = self._transform_steps(X, routed_params)
        with using_output_type("cupy"):
            result = self.steps[-1][1].decision_function(
                Xt,
                **routed_params.get(self.steps[-1][0], {}).get(
                    "decision_function", {}
                ),
            )
        return coerce_arrays(result, output_type)

    def score_samples(self, X):
        output_type = GlobalSettings().output_type or "numpy"
        check_is_fitted(self)
        Xt = self._transform_steps(X)
        with using_output_type("cupy"):
            result = self.steps[-1][1].score_samples(Xt)
        return coerce_arrays(result, output_type)

    def transform(self, X, **params):
        output_type = GlobalSettings().output_type or "numpy"
        check_is_fitted(self)
        _raise_for_params(params, self, "transform")
        routed_params = process_routing(self, "transform", **params)
        Xt = self._transform_steps(X, routed_params)
        with using_output_type("cupy"):
            Xt = self.steps[-1][1].transform(
                Xt, **routed_params[self.steps[-1][0]].transform
            )
        return coerce_arrays(Xt, output_type)

    def inverse_transform(self, X, **params):
        output_type = GlobalSettings().output_type or "numpy"
        check_is_fitted(self)
        _raise_for_params(params, self, "inverse_transform")
        routed_params = process_routing(self, "inverse_transform", **params)
        reverse_iter = reversed(list(self._iter()))
        for step_idx, name, transform in reverse_iter:
            with self._output_type_for_inverse_step(step_idx):
                X = transform.inverse_transform(
                    X, **routed_params[name].inverse_transform
                )
        return coerce_arrays(X, output_type)

    def score(self, X, y=None, sample_weight=None, **params):
        check_is_fitted(self)
        if not _routing_enabled():
            Xt = self._transform_steps(X)
            score_params = {}
            if sample_weight is not None:
                score_params["sample_weight"] = sample_weight
            return self.steps[-1][1].score(Xt, y, **score_params)
        routed_params = process_routing(
            self, "score", sample_weight=sample_weight, **params
        )
        Xt = self._transform_steps(X, routed_params)
        return self.steps[-1][1].score(
            Xt, y, **routed_params[self.steps[-1][0]].score
        )

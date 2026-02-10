#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""Thin Pipeline wrapper for cuml.accel that keeps intermediate data on device."""

from __future__ import annotations

import importlib
from dataclasses import replace
from typing import Any

from sklearn.base import clone
from sklearn.pipeline import Pipeline as _SklearnPipeline
from sklearn.pipeline import _final_estimator_has, _fit_transform_one
from sklearn.utils._tags import TransformerTags
from sklearn.utils._user_interface import _print_elapsed_time
from sklearn.utils.metadata_routing import (
    _raise_for_params,
    _routing_enabled,
    process_routing,
)
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted, check_memory

from cuml.accel import is_proxy
from cuml.accel.estimator_proxy import ProxyBase
from cuml.internals.global_settings import GlobalSettings
from cuml.internals.interop import InteropMixin, UnsupportedOnGPU
from cuml.internals.outputs import coerce_arrays, using_output_type

__all__ = ("Pipeline",)


def _step_estimator_to_cpu(est):
    """Return the CPU (sklearn) estimator for a pipeline step (proxy or plain)."""
    if is_proxy(est):
        est._sync_attrs_to_cpu()
        return est._cpu
    # Non-proxy: already a CPU estimator, fitted in-place by the pipeline
    return est


def _cpu_estimator_to_accel(cpu_est):
    """Return the accel (proxy) estimator for a CPU step when available."""
    try:
        module = importlib.import_module(cpu_est.__class__.__module__)
    except ModuleNotFoundError:
        return cpu_est
    overrides = getattr(module, "_accel_overrides", None)
    if overrides is None:
        return cpu_est
    accel_cls = overrides.get(cpu_est.__class__.__name__)
    if accel_cls is None:
        return cpu_est
    if not hasattr(accel_cls, "_reconstruct_from_cpu"):
        return cpu_est
    try:
        return accel_cls._reconstruct_from_cpu(cpu_est)
    except (UnsupportedOnGPU, TypeError):
        return cpu_est


class _AccelPipeline(_SklearnPipeline, InteropMixin):
    """Device-aware Pipeline implementation used as _gpu_class for the accel proxy.

    Keeps intermediate data on device during fit/predict/transform. Intermediate
    steps output cupy arrays only when the next step is a cuml.accel-wrapped
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
        out = super()._attrs_from_cpu(model)
        # Restore steps: convert each CPU step to accel version where possible
        accel_steps = []
        for name, cpu_est in model.steps:
            if cpu_est is None or cpu_est == "passthrough":
                accel_steps.append((name, cpu_est))
                continue
            accel_est = _cpu_estimator_to_accel(cpu_est)
            accel_steps.append((name, accel_est))
        out["steps"] = accel_steps
        return out

    def _attrs_to_cpu(self, model) -> dict[str, Any]:
        # Only sync steps. sklearn.pipeline.Pipeline has n_features_in_ and
        # feature_names_in_ as read-only properties (delegate to first step);
        # do not try to set them or setattr will raise.
        cpu_steps = []
        for name, est in self.steps:
            if est is None or est == "passthrough":
                cpu_steps.append((name, est))
                continue
            cpu_est = _step_estimator_to_cpu(est)
            cpu_steps.append((name, cpu_est))
        return {"steps": cpu_steps}

    @classmethod
    def from_sklearn(cls, model):
        """Build a device-aware pipeline from a CPU pipeline, converting steps to accel where possible."""
        if not isinstance(model, cls._get_cpu_class()):
            raise TypeError(
                f"Expected instance of {cls._cpu_class_path!r}, got "
                f"{type(model).__name__!r}"
            )
        params = cls._params_from_cpu(model)
        out = cls(**params)
        out._sync_attrs_from_cpu(model)
        if hasattr(out, "output_type"):
            out.output_type = "numpy"
        return out

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

    def _call_final_estimator(self, method_name, X, **params):
        """Transform X through steps, call final estimator's method, coerce output."""
        output_type = GlobalSettings().output_type or "numpy"
        check_is_fitted(self)
        if not _routing_enabled():
            Xt = self._transform_steps(X)
            with using_output_type("cupy"):
                result = getattr(self.steps[-1][1], method_name)(Xt, **params)
        else:
            routed_params = process_routing(self, method_name, **params)
            Xt = self._transform_steps(X, routed_params)
            with using_output_type("cupy"):
                result = getattr(self.steps[-1][1], method_name)(
                    Xt,
                    **getattr(routed_params[self.steps[-1][0]], method_name),
                )
        return coerce_arrays(result, output_type)

    def fit(self, X, y=None, **params):
        return super().fit(X, y, **params)

    @available_if(_SklearnPipeline._can_fit_transform)
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

    @available_if(_final_estimator_has("fit_predict"))
    def fit_predict(self, X, y=None, **params):
        """Fit transformers, then call fit_predict on the final estimator."""
        output_type = GlobalSettings().output_type or "numpy"
        routed_params = self._check_method_params(
            method="fit_predict", props=params
        )
        Xt = self._fit(X, y, routed_params, raw_params=params)
        params_last_step = routed_params[self.steps[-1][0]]
        with _print_elapsed_time(
            "Pipeline", self._log_message(len(self.steps) - 1)
        ):
            with using_output_type("cupy"):
                y_pred = self.steps[-1][1].fit_predict(
                    Xt, y, **params_last_step.get("fit_predict", {})
                )
        return coerce_arrays(y_pred, output_type)

    @available_if(_final_estimator_has("predict"))
    def predict(self, X, **params):
        return self._call_final_estimator("predict", X, **params)

    @available_if(_final_estimator_has("predict_proba"))
    def predict_proba(self, X, **params):
        return self._call_final_estimator("predict_proba", X, **params)

    @available_if(_final_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X, **params):
        return self._call_final_estimator("predict_log_proba", X, **params)

    @available_if(_final_estimator_has("decision_function"))
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

    @available_if(_final_estimator_has("score_samples"))
    def score_samples(self, X):
        output_type = GlobalSettings().output_type or "numpy"
        check_is_fitted(self)
        Xt = self._transform_steps(X)
        with using_output_type("cupy"):
            result = self.steps[-1][1].score_samples(Xt)
        return coerce_arrays(result, output_type)

    @available_if(_SklearnPipeline._can_transform)
    def transform(self, X, **params):
        output_type = GlobalSettings().output_type or "numpy"
        check_is_fitted(self)
        _raise_for_params(params, self, "transform")
        routed_params = process_routing(self, "transform", **params)
        Xt = self._transform_steps(X, routed_params)
        last_step = self.steps[-1][1]
        if last_step is not None and last_step != "passthrough":
            with using_output_type("cupy"):
                Xt = last_step.transform(
                    Xt, **routed_params[self.steps[-1][0]].transform
                )
        return coerce_arrays(Xt, output_type)

    @available_if(_SklearnPipeline._can_inverse_transform)
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

    @available_if(_final_estimator_has("score"))
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


class Pipeline(ProxyBase):
    """cuml.accel proxy for sklearn.pipeline.Pipeline.

    Keeps intermediate data on device when possible. Subclasses ProxyBase like
    other accel estimators; the device-aware implementation is _AccelPipeline.
    """

    _gpu_class = _AccelPipeline

    # _AccelPipeline is a sklearn Pipeline subclass, not a cuML estimator;
    # it has no _get_tags()["X_types_gpu"]. Override so ProxyBase does not expect it.
    _gpu_supports_sparse = False
    # Access to steps/named_steps must sync fitted state from _gpu so step estimators
    # (and their fitted attributes) are available on _cpu.
    _other_attributes = frozenset({"steps", "named_steps"})

    def __sklearn_tags__(self):
        # sklearn's Pipeline.__sklearn_tags__() can leave transformer_tags=None
        # when the last step is not a transformer or steps are not yet set.
        # Common tests require transformer_tags to be set for any estimator
        # with transform(). Override __sklearn_tags__ (not just _get_tags)
        # because get_tags() calls __sklearn_tags__, not _get_tags.
        tags = self._cpu.__sklearn_tags__()
        if tags.transformer_tags is None:
            return replace(tags, transformer_tags=TransformerTags())
        return tags

    def __getitem__(self, ind):
        """Return a sub-pipeline or a single estimator; delegate to CPU and wrap slices."""
        self._sync_attrs_to_cpu()
        result = self._cpu[ind]
        if isinstance(result, _SklearnPipeline):
            return type(self)._reconstruct_from_cpu(result)
        return result

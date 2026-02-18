#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""Thin Pipeline wrapper for cuml.accel that keeps intermediate data on device."""

from __future__ import annotations

from dataclasses import replace
from functools import wraps

from sklearn.pipeline import Pipeline as _SklearnPipeline
from sklearn.utils._tags import TransformerTags
from sklearn.utils.metaestimators import available_if

from cuml._thirdparty.sklearn.preprocessing._pipeline import (
    Pipeline as _AccelPipeline,
)
from cuml.accel.estimator_proxy import ProxyBase

__all__ = ("Pipeline", "make_pipeline")


def _cpu_has(method_name):
    """Check that the CPU Pipeline supports a conditional method.

    Uses getattr so that when the method is not available, the CPU pipeline's
    AttributeError (and its __cause__, e.g. from the final estimator) is
    re-raised. The proxy's @available_if descriptor then chains from it,
    giving uniform exception chaining for all conditional methods (transform,
    fit_transform, fit_predict, predict, predict_proba, predict_log_proba,
    decision_function, score_samples, inverse_transform, score).
    """

    def check(self):
        try:
            getattr(self._cpu, method_name)
            return True
        except AttributeError as e:
            cause = e.__cause__ if e.__cause__ is not None else e
            raise cause

    return check


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

    # Conditional methods: use @available_if to mirror sklearn Pipeline's
    # conditional availability, and @wraps to inherit the sklearn docstrings
    # so numpydoc validation passes.

    @available_if(_cpu_has("transform"))
    @wraps(_SklearnPipeline.transform, assigned=("__doc__",), updated=())
    def transform(self, X, **params):
        return self._call_method("transform", X, **params)

    @available_if(_cpu_has("fit_transform"))
    @wraps(_SklearnPipeline.fit_transform, assigned=("__doc__",), updated=())
    def fit_transform(self, X, y=None, **params):
        return self._call_method("fit_transform", X, y, **params)

    @available_if(_cpu_has("fit_predict"))
    @wraps(_SklearnPipeline.fit_predict, assigned=("__doc__",), updated=())
    def fit_predict(self, X, y=None, **params):
        return self._call_method("fit_predict", X, y, **params)

    @available_if(_cpu_has("predict"))
    @wraps(_SklearnPipeline.predict, assigned=("__doc__",), updated=())
    def predict(self, X, **params):
        return self._call_method("predict", X, **params)

    @available_if(_cpu_has("predict_proba"))
    @wraps(_SklearnPipeline.predict_proba, assigned=("__doc__",), updated=())
    def predict_proba(self, X, **params):
        return self._call_method("predict_proba", X, **params)

    @available_if(_cpu_has("predict_log_proba"))
    @wraps(
        _SklearnPipeline.predict_log_proba, assigned=("__doc__",), updated=()
    )
    def predict_log_proba(self, X, **params):
        return self._call_method("predict_log_proba", X, **params)

    @available_if(_cpu_has("decision_function"))
    @wraps(
        _SklearnPipeline.decision_function, assigned=("__doc__",), updated=()
    )
    def decision_function(self, X, **params):
        return self._call_method("decision_function", X, **params)

    @available_if(_cpu_has("score_samples"))
    @wraps(_SklearnPipeline.score_samples, assigned=("__doc__",), updated=())
    def score_samples(self, X):
        return self._call_method("score_samples", X)

    @available_if(_cpu_has("inverse_transform"))
    @wraps(
        _SklearnPipeline.inverse_transform, assigned=("__doc__",), updated=()
    )
    def inverse_transform(self, X, **params):
        return self._call_method("inverse_transform", X, **params)

    @available_if(_cpu_has("score"))
    @wraps(_SklearnPipeline.score, assigned=("__doc__",), updated=())
    def score(self, X, y=None, sample_weight=None, **params):
        return self._call_method(
            "score", X, y, sample_weight=sample_weight, **params
        )

    def __sklearn_tags__(self):
        # sklearn's Pipeline.__sklearn_tags__() can leave transformer_tags=None
        # when the last step is not a transformer or steps are not yet set.
        # Common tests require transformer_tags to be set for any estimator
        # with transform(). Override __sklearn_tags__ (not just _get_tags)
        # because get_tags() calls __sklearn_tags__, not _get_tags.
        tags = self._cpu.__sklearn_tags__()
        if tags.transformer_tags is None and hasattr(self._cpu, "transform"):
            return replace(tags, transformer_tags=TransformerTags())
        return tags

    def __getitem__(self, ind):
        """Return a sub-pipeline or a single estimator; delegate to CPU and wrap slices."""
        self._sync_attrs_to_cpu()
        result = self._cpu[ind]
        if isinstance(result, _SklearnPipeline):
            return type(self)._reconstruct_from_cpu(result)
        return result


def make_pipeline(*steps, memory=None, transform_input=None, verbose=False):
    """Construct a Pipeline from the given estimators (cuml.accel version).

    Delegates to sklearn's _name_estimators then builds our Pipeline so that
    make_pipeline(...) returns an accelerated Pipeline instance.
    """
    from sklearn.pipeline import _name_estimators

    return Pipeline(
        _name_estimators(steps),
        memory=memory,
        transform_input=transform_input,
        verbose=verbose,
    )

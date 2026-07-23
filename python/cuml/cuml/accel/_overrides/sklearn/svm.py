# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import functools

import numpy as np
from sklearn.svm import SVC as _SVC
from sklearn.utils.metaestimators import available_if

import cuml.svm
from cuml.accel.estimator_proxy import ProxyBase
from cuml.internals.interop import UnsupportedOnGPU

__all__ = (
    "SVC",
    "SVR",
    "LinearSVC",
    "LinearSVR",
)


def _has_probability(model):
    # sklearn >= 1.9 defaults `probability` to the "deprecated" sentinel,
    # which is truthy. Treat it like False (no calibration requested).
    if model.probability == "deprecated" or not model.probability:
        raise AttributeError(
            "predict_proba is not available when probability=False"
        )
    return True


class SVC(ProxyBase):
    _gpu_class = cuml.svm.SVC
    _other_attributes = frozenset(("_gamma",))

    def _gpu_fit(self, X, y, sample_weight=None):
        classes = np.unique(np.asanyarray(y))
        if len(classes) > 2:
            raise UnsupportedOnGPU("Multiclass `y` is not supported")
        return self._gpu.fit(X, y, sample_weight=sample_weight)

    def _gpu_decision_function(self, X):
        # Fixup returned dtype
        return self._gpu.decision_function(X).astype("float64", copy=False)

    # Manual gate: ProxyBase has no built-in conditional-method support.
    @available_if(_has_probability)
    @functools.wraps(_SVC.predict_proba)
    def predict_proba(self, X):
        return self._call_method("predict_proba", X)

    @available_if(_has_probability)
    @functools.wraps(_SVC.predict_log_proba)
    def predict_log_proba(self, X):
        return self._call_method("predict_log_proba", X)


class SVR(ProxyBase):
    _gpu_class = cuml.svm.SVR
    _other_attributes = frozenset(("_gamma",))


class LinearSVC(ProxyBase):
    _gpu_class = cuml.svm.LinearSVC


class LinearSVR(ProxyBase):
    _gpu_class = cuml.svm.LinearSVR

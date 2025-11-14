# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
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
    if not model.probability:
        raise AttributeError(
            "predict_proba is not available when probability=False"
        )
    return True


class SVC(ProxyBase):
    _gpu_class = cuml.svm.SVC
    _not_implemented_attributes = frozenset(
        (
            "class_weight_",
            "n_iter_",
        )
    )

    def _gpu_fit(self, X, y, sample_weight=None):
        classes, counts = np.unique(np.asanyarray(y), return_counts=True)
        if len(classes) > 2:
            raise UnsupportedOnGPU("Multiclass `y` is not supported")

        # CalibratedClassifierCV doesn't like working with cases where any
        # classes have less than 5 examples.
        if self.probability and counts.min() < 5:
            raise UnsupportedOnGPU(
                "`probability=True` requires >= 5 samples per class"
            )

        return self._gpu.fit(X, y, sample_weight=sample_weight)

    def _gpu_decision_function(self, X):
        # Fixup returned dtype
        return self._gpu.decision_function(X).astype("float64", copy=False)

    # XXX: sklearn wants these methods to only exist if probability=True.
    # ProxyBase lacks a builtin mechanism to do that, since this is the only
    # use case so far we manually define them for now.
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
    _not_implemented_attributes = frozenset(("n_iter_",))


class LinearSVC(ProxyBase):
    _gpu_class = cuml.svm.LinearSVC


class LinearSVR(ProxyBase):
    _gpu_class = cuml.svm.LinearSVR

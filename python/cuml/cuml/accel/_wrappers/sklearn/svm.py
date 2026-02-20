# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import cuml.svm
from cuml.accel.estimator_proxy import ProxyBase
from cuml.internals.interop import UnsupportedOnGPU

__all__ = (
    "SVC",
    "SVR",
    "LinearSVC",
    "LinearSVR",
)


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


class SVR(ProxyBase):
    _gpu_class = cuml.svm.SVR
    _not_implemented_attributes = frozenset(("n_iter_",))


class LinearSVC(ProxyBase):
    _gpu_class = cuml.svm.LinearSVC


class LinearSVR(ProxyBase):
    _gpu_class = cuml.svm.LinearSVR

# Copyright (c) 2025, NVIDIA CORPORATION.
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
    # cuml.SVC supports sparse X for some but not all operations,
    # easier to just fallback for now
    _gpu_supports_sparse = False
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
    # cuml.SVC supports sparse X for some but not all operations,
    # easier to just fallback for now
    _gpu_supports_sparse = False
    _not_implemented_attributes = frozenset(("n_iter_",))


class LinearSVC(ProxyBase):
    _gpu_class = cuml.svm.LinearSVC
    _not_implemented_attributes = frozenset(("n_iter_",))


class LinearSVR(ProxyBase):
    _gpu_class = cuml.svm.LinearSVR
    _not_implemented_attributes = frozenset(("n_iter_",))

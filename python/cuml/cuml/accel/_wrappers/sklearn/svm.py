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
import sklearn.svm
from sklearn.utils.metaestimators import available_if

import cuml.svm
from cuml.accel.estimator_proxy import ProxyBase
from cuml.internals.interop import UnsupportedOnGPU

__all__ = (
    "SVC",
    "SVR",
)


def _has_probability(model):
    if not model.probability:
        raise AttributeError(
            "predict_proba is not available when probability=False"
        )


class SVC(ProxyBase):
    _gpu_class = cuml.svm.SVC
    # cuml.SVC supports sparse X for some but not all operations,
    # easier to just fallback for now
    _gpu_supports_sparse = False

    def _gpu_fit(self, X, y, sample_weight=None):
        n_classes = len(np.unique(np.asanyarray(y)))
        if n_classes > 2:
            raise UnsupportedOnGPU("Multiclass `y` is not supported")
        return self._gpu.fit(X, y, sample_weight=sample_weight)

    def _gpu_decision_function(self, X):
        # Fixup returned dtype
        return self._gpu.decision_function(X).astype("float64", copy=False)

    # XXX: sklearn wants these methods to only exist if probability=True.
    # ProxyBase lacks a builtin mechanism to do that, since this is the only
    # use case so far we manually define them for now.
    @available_if(_has_probability)
    @functools.wraps(sklearn.svm.SVC.predict_proba)
    def predict_proba(self, X):
        return self._call_method("predict_proba", X)

    @available_if(_has_probability)
    @functools.wraps(sklearn.svm.SVC.predict_log_proba)
    def predict_log_proba(self, X):
        return self._call_method("predict_log_proba", X)


class SVR(ProxyBase):
    _gpu_class = cuml.svm.SVR
    # cuml.SVC supports sparse X for some but not all operations,
    # easier to just fallback for now
    _gpu_supports_sparse = False

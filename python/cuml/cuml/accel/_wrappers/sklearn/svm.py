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

import numpy as np

import cuml.svm
from cuml.accel.estimator_proxy import ProxyBase
from cuml.internals.interop import UnsupportedOnGPU

__all__ = (
    "SVC",
    "SVR",
)


class SVC(ProxyBase):
    _gpu_class = cuml.svm.SVC
    # cuml.SVC supports sparse X for some but not all operations,
    # easier to just fallback for now
    _gpu_supports_sparse = False

    def _gpu_fit(self, X, y, sample_weight=None):
        n_classes = len(np.unique(np.asanyarray(y)))
        if n_classes > 2:
            raise UnsupportedOnGPU("SVC.fit doesn't support multiclass")
        return self._gpu.fit(X, y, sample_weight=sample_weight)


class SVR(ProxyBase):
    _gpu_class = cuml.svm.SVR
    # cuml.SVC supports sparse X for some but not all operations,
    # easier to just fallback for now
    _gpu_supports_sparse = False

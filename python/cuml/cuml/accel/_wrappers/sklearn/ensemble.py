#
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
#

import cuml.ensemble
from cuml.accel.estimator_proxy import ProxyBase
from cuml.internals.interop import UnsupportedOnGPU

__all__ = ("RandomForestRegressor", "RandomForestClassifier")


class RandomForestRegressor(ProxyBase):
    _gpu_class = cuml.ensemble.RandomForestRegressor

    def _gpu_fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise UnsupportedOnGPU("`sample_weight` is not supported")
        return self._gpu.fit(X, y)

    def _gpu_score(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise UnsupportedOnGPU("`sample_weight` is not supported")
        return self._gpu.score(X, y)

    def __len__(self):
        return self._call_method("__len__")

    def __iter__(self):
        return self._call_method("__iter__")

    def __getitem__(self, index):
        return self._call_method("__getitem__", index)


class RandomForestClassifier(ProxyBase):
    _gpu_class = cuml.ensemble.RandomForestClassifier

    def _gpu_fit(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise UnsupportedOnGPU("`sample_weight` is not supported")
        return self._gpu.fit(X, y)

    def _gpu_score(self, X, y, sample_weight=None):
        if sample_weight is not None:
            raise UnsupportedOnGPU("`sample_weight` is not supported")
        return self._gpu.score(X, y)

    def __len__(self):
        return self._call_method("__len__")

    def __iter__(self):
        return self._call_method("__iter__")

    def __getitem__(self, index):
        return self._call_method("__getitem__", index)

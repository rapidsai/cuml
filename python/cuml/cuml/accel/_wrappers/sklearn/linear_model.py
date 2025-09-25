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

import cuml.linear_model
from cuml.accel.estimator_proxy import ProxyBase
from cuml.internals.input_utils import input_to_cuml_array
from cuml.internals.interop import UnsupportedOnGPU

__all__ = (
    "LinearRegression",
    "LogisticRegression",
    "ElasticNet",
    "Ridge",
    "Lasso",
)


class LinearRegression(ProxyBase):
    _gpu_class = cuml.linear_model.LinearRegression
    _not_implemented_attributes = frozenset(("rank_", "singular_"))


class LogisticRegression(ProxyBase):
    _gpu_class = cuml.linear_model.LogisticRegression


class ElasticNet(ProxyBase):
    _gpu_class = cuml.linear_model.ElasticNet
    _not_implemented_attributes = frozenset(("dual_gap_", "n_iter_"))

    def _gpu_fit(self, X, y, sample_weight=None, check_input=True):
        # Fixes signature mismatch with cuml.ElasticNet. check_input can be ignored.
        return self._gpu.fit(X, y, sample_weight=sample_weight)


class Ridge(ProxyBase):
    _gpu_class = cuml.linear_model.Ridge
    _not_implemented_attributes = frozenset(("n_iter_",))

    def _gpu_fit(self, X, y, sample_weight=None):
        X = input_to_cuml_array(X, convert_to_mem_type=False)[0]
        y = input_to_cuml_array(y, convert_to_mem_type=False)[0]
        if len(y.shape) > 1:
            raise UnsupportedOnGPU("Multioutput `y` is not supported")

        if X.shape[0] < X.shape[1]:
            raise UnsupportedOnGPU(
                "`X` with more columns than rows is not supported"
            )

        return self._gpu.fit(X, y, sample_weight=sample_weight)


class Lasso(ProxyBase):
    _gpu_class = cuml.linear_model.Lasso
    _not_implemented_attributes = frozenset(("dual_gap_", "n_iter_"))

    def _gpu_fit(self, X, y, sample_weight=None, check_input=True):
        # Fixes signature mismatch with cuml.Lasso. check_input can be ignored.
        return self._gpu.fit(X, y, sample_weight=sample_weight)

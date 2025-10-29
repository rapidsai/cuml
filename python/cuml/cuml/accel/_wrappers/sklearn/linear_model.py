#
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cuml.linear_model
from cuml.accel.estimator_proxy import ProxyBase

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


class Lasso(ProxyBase):
    _gpu_class = cuml.linear_model.Lasso
    _not_implemented_attributes = frozenset(("dual_gap_", "n_iter_"))

    def _gpu_fit(self, X, y, sample_weight=None, check_input=True):
        # Fixes signature mismatch with cuml.Lasso. check_input can be ignored.
        return self._gpu.fit(X, y, sample_weight=sample_weight)

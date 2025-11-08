#
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import sklearn
from packaging.version import Version

import cuml.linear_model
from cuml.accel.estimator_proxy import ProxyBase
from cuml.internals.array import CumlArray
from cuml.internals.memory_utils import using_output_type

__all__ = (
    "LinearRegression",
    "LogisticRegression",
    "ElasticNet",
    "Ridge",
    "Lasso",
)


SKLEARN_16 = Version(sklearn.__version__) >= Version("1.6.0")


class LinearRegression(ProxyBase):
    _gpu_class = cuml.linear_model.LinearRegression
    _not_implemented_attributes = frozenset(("rank_", "singular_"))


class LogisticRegression(ProxyBase):
    _gpu_class = cuml.linear_model.LogisticRegression


class ElasticNet(ProxyBase):
    _gpu_class = cuml.linear_model.ElasticNet
    _not_implemented_attributes = frozenset(("dual_gap_",))

    def _gpu_fit(self, X, y, sample_weight=None, check_input=True):
        # Fixes signature mismatch with cuml.ElasticNet. check_input can be ignored.
        return self._gpu.fit(X, y, sample_weight=sample_weight)


class Ridge(ProxyBase):
    _gpu_class = cuml.linear_model.Ridge
    _not_implemented_attributes = frozenset(("n_iter_",))

    def _gpu_fit(self, X, y, sample_weight=None):
        self._gpu.fit(X, y, sample_weight=sample_weight)

        # XXX: sklearn 1.6 changed the shape of `coef_` when fit with a 1
        # column 2D y. The sklearn 1.6+ behavior is what we implement in
        # cuml.Ridge, here we adjust the shape of `coef_` after the fit to
        # match the older behavior. This will also trickle down to change the
        # output shape of `predict` to match the older behavior transparently.
        if not SKLEARN_16 and (y_shape := getattr(y, "shape", ())):
            if len(y_shape) == 2 and y_shape[1] == 1:
                with using_output_type("cupy"):
                    # Reshape coef_ to be a 2D array
                    self._gpu.coef_ = CumlArray(
                        data=self._gpu.coef_.reshape(1, -1)
                    )

        return self


class Lasso(ProxyBase):
    _gpu_class = cuml.linear_model.Lasso
    _not_implemented_attributes = frozenset(("dual_gap_",))

    def _gpu_fit(self, X, y, sample_weight=None, check_input=True):
        # Fixes signature mismatch with cuml.Lasso. check_input can be ignored.
        return self._gpu.fit(X, y, sample_weight=sample_weight)

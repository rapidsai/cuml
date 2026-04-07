#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import sklearn
from packaging.version import Version

import cuml.linear_model
from cuml.accel.estimator_proxy import ProxyBase
from cuml.common.sparse_utils import is_sparse
from cuml.internals.array import CumlArray
from cuml.internals.input_utils import input_to_cuml_array
from cuml.internals.interop import UnsupportedOnGPU
from cuml.internals.outputs import using_output_type

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


class Ridge(ProxyBase):
    _gpu_class = cuml.linear_model.Ridge

    def _gpu_fit(self, X, y, sample_weight=None):
        if is_sparse(X) and (
            self.solver not in ("auto", "lsqr", "lbfgs", "sparse_cg")
        ):
            # cuml.Ridge's sparse solver is iterative. For maximum compatibility,
            # we only proxy through iterative sparse solvers.
            raise UnsupportedOnGPU(
                f"Sparse inputs are not supported with solver={self.solver!r}"
            )

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


class ElasticNet(ProxyBase):
    _gpu_class = cuml.linear_model.ElasticNet
    _not_implemented_attributes = frozenset(("dual_gap_",))

    def _gpu_fit(self, X, y, sample_weight=None, check_input=True):
        # check_input is ignored, only here to fix signature mismatch with sklearn

        y = input_to_cuml_array(y, convert_to_mem_type=False)[0]
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise UnsupportedOnGPU("Multi-output targets are not supported")
        return self._gpu.fit(X, y, sample_weight=sample_weight)


class Lasso(ProxyBase):
    _gpu_class = cuml.linear_model.Lasso
    _not_implemented_attributes = frozenset(("dual_gap_",))

    def _gpu_fit(self, X, y, sample_weight=None, check_input=True):
        # check_input is ignored, only here to fix signature mismatch with sklearn

        y = input_to_cuml_array(y, convert_to_mem_type=False)[0]
        if len(y.shape) > 1 and y.shape[1] > 1:
            raise UnsupportedOnGPU("Multi-output targets are not supported")
        return self._gpu.fit(X, y, sample_weight=sample_weight)

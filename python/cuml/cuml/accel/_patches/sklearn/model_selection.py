# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import functools

import cupy as cp
import scipy.sparse
import sklearn
from packaging.version import Version
from sklearn.model_selection import GridSearchCV, ParameterGrid

from cuml.accel.estimator_proxy import ensure_host, is_proxy
from cuml.internals.global_settings import GlobalSettings
from cuml.internals.interop import UnsupportedOnGPU
from cuml.internals.outputs import using_output_type

AT_LEAST_SKLEARN_16 = Version(sklearn.__version__) >= Version("1.6.0")

__all__ = ("GridSearchCV",)


def _patch_fit(cls):
    orig_fit = cls.fit

    @functools.wraps(orig_fit)
    def fit(self, X, y=None, **params):
        if (
            not AT_LEAST_SKLEARN_16
            or not is_proxy(self.estimator)
            or scipy.sparse.issparse(X)
        ):
            return orig_fit(self, X, y, **params)

        # Pre-check: does any param combination support GPU?
        gpu_class = type(self.estimator)._gpu_class
        any_gpu = False
        for candidate in ParameterGrid(self.param_grid):
            try:
                cpu_clone = sklearn.clone(self.estimator._cpu).set_params(
                    **candidate
                )
                gpu_class._params_from_cpu(cpu_clone)
                any_gpu = True
                break
            except UnsupportedOnGPU:
                continue
        if not any_gpu:
            return orig_fit(self, X, y, **params)

        X_gpu = cp.asarray(X) if not isinstance(X, cp.ndarray) else X
        y_gpu = (
            cp.asarray(y)
            if y is not None and not isinstance(y, cp.ndarray)
            else y
        )

        orig_n_jobs = self.n_jobs
        if self.n_jobs is not None and self.n_jobs != 1:
            self.n_jobs = 1

        try:
            with (
                sklearn.config_context(array_api_dispatch=True),
                using_output_type("cupy"),
            ):
                out = orig_fit(self, X_gpu, y_gpu, **params)
        finally:
            self.n_jobs = orig_n_jobs

        # Ensure user-facing attributes are host arrays
        if GlobalSettings().output_type in (None, "numpy"):
            for attr in ("best_score_", "best_index_"):
                val = getattr(self, attr, None)
                if val is not None:
                    setattr(self, attr, ensure_host(val))

        return out

    cls.fit = fit


_patch_fit(GridSearchCV)

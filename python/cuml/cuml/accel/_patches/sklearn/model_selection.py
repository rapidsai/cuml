# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import contextlib
import functools
import os

import cupy as cp
import numpy as np
import scipy.sparse
import sklearn
from packaging.version import Version
from sklearn.model_selection import GridSearchCV, ParameterGrid

from cuml.accel.core import logger
from cuml.accel.estimator_proxy import ensure_host, is_proxy
from cuml.internals.global_settings import GlobalSettings
from cuml.internals.interop import UnsupportedOnGPU
from cuml.internals.outputs import using_output_type

AT_LEAST_SKLEARN_18 = Version(sklearn.__version__) >= Version("1.8.0")

__all__ = ("GridSearchCV",)


@contextlib.contextmanager
def _enable_scipy_array_api():
    """Enable scipy's array API support.

    Sets the SCIPY_ARRAY_API env var (checked by sklearn's config validation)
    and updates scipy's cached config (in case scipy had already been imported).

    Both are restored on exit.
    """
    old_env = os.environ.get("SCIPY_ARRAY_API")
    os.environ["SCIPY_ARRAY_API"] = "1"

    old_cached = None
    try:
        # XXX What is the oldest scipy version that we might import
        # XXX do all of them have this?
        import scipy._lib._array_api as _sa

        old_cached = _sa._GLOBAL_CONFIG.get("SCIPY_ARRAY_API")
        _sa._GLOBAL_CONFIG["SCIPY_ARRAY_API"] = "1"
    except (ImportError, AttributeError):
        _sa = None

    try:
        yield
    finally:
        if old_env is None:
            os.environ.pop("SCIPY_ARRAY_API", None)
        else:
            os.environ["SCIPY_ARRAY_API"] = old_env

        if _sa is not None and old_cached is not None:
            _sa._GLOBAL_CONFIG["SCIPY_ARRAY_API"] = old_cached


def _patch_fit(cls):
    orig_fit = cls.fit

    @functools.wraps(orig_fit)
    def fit(self, X, y=None, **params):
        estimator_name = type(self.estimator).__name__

        if not AT_LEAST_SKLEARN_18:
            logger.debug(
                "`GridSearchCV.fit` not optimized: requires sklearn >= 1.8"
            )
            return orig_fit(self, X, y, **params)

        if not is_proxy(self.estimator):
            logger.info(
                f"`GridSearchCV.fit` not optimized: "
                f"`{estimator_name}` is not a cuml.accel proxy"
            )
            return orig_fit(self, X, y, **params)

        if scipy.sparse.issparse(X):
            logger.info("`GridSearchCV.fit` not optimized: sparse input")
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
            logger.info(
                f"`GridSearchCV.fit` not optimized: no parameter "
                f"combinations in the grid support GPU for `{estimator_name}`"
            )
            return orig_fit(self, X, y, **params)

        logger.info(
            f"`GridSearchCV.fit` input data moved to GPU as some "
            f"parameter combinations support acceleration for "
            f"`{estimator_name}`"
        )

        X_gpu = cp.asarray(X) if not isinstance(X, cp.ndarray) else X
        y_gpu = (
            cp.asarray(y)
            if y is not None and not isinstance(y, cp.ndarray)
            else y
        )
        # Convert array-like params (e.g. sample_weight) to cupy so scoring
        # metrics see consistent array types. Exclude "groups" which goes to
        # the CV splitter and must stay on host.
        params = {
            k: cp.asarray(v)
            if isinstance(v, np.ndarray) and k != "groups"
            else v
            for k, v in params.items()
        }

        orig_n_jobs = self.n_jobs
        if self.n_jobs is not None and self.n_jobs != 1:
            logger.info(
                f"`GridSearchCV.fit` forcing n_jobs=1 (was {self.n_jobs}) "
                f"for GPU execution"
            )
            self.n_jobs = 1

        try:
            with (
                _enable_scipy_array_api(),
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

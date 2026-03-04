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
from sklearn.pipeline import Pipeline

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


def _contains_proxy(estimator):
    """Check if an estimator can benefit from the cupy data path.

    For bare proxies this is always True. For Pipelines, we reuse the
    Pipeline patch's ``get_output_type`` which returns ``"cupy"`` only
    when there is an unbroken chain of proxy steps from the first proxy
    to the end. If the Pipeline would return numpy predictions, the cupy
    y_test from array-API CV splitting would cause a device mismatch in
    scoring.
    """
    if is_proxy(estimator):
        return True
    if isinstance(estimator, Pipeline):
        from cuml.accel._patches.sklearn.pipeline import get_output_type

        return get_output_type(estimator) == "cupy"
    return False


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

        if not _contains_proxy(self.estimator):
            logger.debug(
                f"`GridSearchCV.fit` not optimized: "
                f"`{estimator_name}` does not contain accelerated estimators"
            )
            return orig_fit(self, X, y, **params)

        if scipy.sparse.issparse(X):
            logger.debug("`GridSearchCV.fit` not optimized: sparse input")
            return orig_fit(self, X, y, **params)

        if y is not None and np.asarray(y).dtype.kind not in "fiub":
            logger.debug("`GridSearchCV.fit` not optimized: non-numeric y")
            return orig_fit(self, X, y, **params)

        if self.n_jobs is not None and self.n_jobs != 1:
            logger.debug(
                f"`GridSearchCV.fit` not optimized: n_jobs={self.n_jobs} "
                f"(set n_jobs=1 for GPU acceleration)"
            )
            return orig_fit(self, X, y, **params)

        # Pre-check for bare proxies: does any param combination support GPU?
        # For Pipelines this is skipped -- the Pipeline patch handles
        # per-step fallback internally.
        if is_proxy(self.estimator):
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
                logger.debug(
                    f"`GridSearchCV.fit` not optimized: no parameter "
                    f"combinations in the grid support GPU for "
                    f"`{estimator_name}`"
                )
                return orig_fit(self, X, y, **params)

        logger.debug(
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

        with (
            _enable_scipy_array_api(),
            sklearn.config_context(array_api_dispatch=True),
            using_output_type("cupy"),
        ):
            out = orig_fit(self, X_gpu, y_gpu, **params)

        # Ensure user-facing attributes are host arrays
        if GlobalSettings().output_type in (None, "numpy"):
            for attr in ("best_score_", "best_index_"):
                val = getattr(self, attr, None)
                if val is not None:
                    setattr(self, attr, ensure_host(val))

        return out

    cls.fit = fit


_patch_fit(GridSearchCV)

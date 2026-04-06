# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import contextlib
import functools

import cupy as cp
import numpy as np
import scipy.sparse
import sklearn
from sklearn.preprocessing import StandardScaler

from cuml.accel._patches.sklearn._utils import enable_scipy_array_api
from cuml.accel.core import logger
from cuml.accel.estimator_proxy import ensure_host
from cuml.internals.global_settings import GlobalSettings
from cuml.internals.outputs import using_output_type

__all__ = ("StandardScaler",)

_FITTED_ATTRS = ("mean_", "var_", "scale_", "n_samples_seen_")


def _can_accelerate(X):
    """Check if X is suitable for GPU acceleration."""
    if scipy.sparse.issparse(X):
        return False
    if hasattr(X, "dtype"):
        if np.issubdtype(X.dtype, np.complexfloating):
            return False
        if X.dtype == np.object_:
            return False
        if X.dtype == np.float16:
            return False
    return True


def _ensure_fitted_on_host(self):
    """Convert fitted attributes to host arrays when output_type is numpy."""
    if GlobalSettings().output_type in (None, "numpy"):
        for attr in _FITTED_ATTRS:
            val = getattr(self, attr, None)
            if val is not None:
                setattr(self, attr, ensure_host(val))


def _to_cupy(X):
    return X if isinstance(X, cp.ndarray) else cp.asarray(X)


def _promote_fitted_to_device(self):
    """Convert any numpy fitted attributes to cupy in-place.

    After fit, fitted attrs are stored as numpy (for user convenience).
    sklearn's array-API code path needs them in the same namespace as
    the cupy input.
    """
    for attr in _FITTED_ATTRS:
        val = getattr(self, attr, None)
        if val is not None and isinstance(val, np.ndarray):
            setattr(self, attr, cp.asarray(val))


@contextlib.contextmanager
def _fitted_attrs_on_device(self):
    """Temporarily move fitted attributes to cupy for a computation.

    Use this for read-only methods (transform, inverse_transform) that
    should not permanently alter the stored attributes.  For methods
    that update attributes (partial_fit) call ``_promote_fitted_to_device``
    directly and let ``_ensure_fitted_on_host`` convert back afterward.
    """
    saved = {}
    for attr in _FITTED_ATTRS:
        val = getattr(self, attr, None)
        if val is not None and isinstance(val, np.ndarray):
            saved[attr] = val
            setattr(self, attr, cp.asarray(val))
    try:
        yield
    finally:
        for attr, val in saved.items():
            setattr(self, attr, val)


def _patch_fit(cls):
    orig_fit = cls.fit

    @functools.wraps(orig_fit)
    def fit(self, X, y=None, sample_weight=None):
        if sample_weight is not None:
            logger.debug("`StandardScaler.fit` not optimized: sample_weight")
            return orig_fit(self, X, y, sample_weight=sample_weight)

        if not _can_accelerate(X):
            logger.debug(
                "`StandardScaler.fit` not optimized: unsupported input"
            )
            return orig_fit(self, X, y)

        logger.debug("`StandardScaler.fit` input data moved to GPU")

        with (
            enable_scipy_array_api(),
            sklearn.config_context(array_api_dispatch=True),
            using_output_type("cupy"),
        ):
            out = orig_fit(self, _to_cupy(X), y)

        _ensure_fitted_on_host(self)
        return out

    cls.fit = fit


def _patch_partial_fit(cls):
    orig_partial_fit = cls.partial_fit

    @functools.wraps(orig_partial_fit)
    def partial_fit(self, X, y=None, sample_weight=None):
        if sample_weight is not None:
            logger.debug(
                "`StandardScaler.partial_fit` not optimized: sample_weight"
            )
            return orig_partial_fit(self, X, y, sample_weight=sample_weight)

        if not _can_accelerate(X):
            logger.debug(
                "`StandardScaler.partial_fit` not optimized: unsupported input"
            )
            return orig_partial_fit(self, X, y)

        logger.debug("`StandardScaler.partial_fit` input data moved to GPU")

        _promote_fitted_to_device(self)
        with (
            enable_scipy_array_api(),
            sklearn.config_context(array_api_dispatch=True),
            using_output_type("cupy"),
        ):
            out = orig_partial_fit(self, _to_cupy(X), y)

        _ensure_fitted_on_host(self)
        return out

    cls.partial_fit = partial_fit


def _patch_transform(cls):
    orig_transform = cls.transform

    @functools.wraps(orig_transform)
    def transform(self, X, copy=None):
        if not _can_accelerate(X):
            logger.debug(
                "`StandardScaler.transform` not optimized: unsupported input"
            )
            return orig_transform(self, X, copy=copy)

        logger.debug("`StandardScaler.transform` input data moved to GPU")

        with (
            _fitted_attrs_on_device(self),
            enable_scipy_array_api(),
            sklearn.config_context(array_api_dispatch=True),
            using_output_type("cupy"),
        ):
            out = orig_transform(self, _to_cupy(X), copy=copy)

        if GlobalSettings().output_type in (None, "numpy"):
            out = ensure_host(out)

        return out

    cls.transform = transform


def _patch_fit_transform(cls):
    orig_fit_transform = cls.fit_transform

    @functools.wraps(orig_fit_transform)
    def fit_transform(self, X, y=None, **fit_params):
        if fit_params.get("sample_weight") is not None:
            logger.debug(
                "`StandardScaler.fit_transform` not optimized: sample_weight"
            )
            return orig_fit_transform(self, X, y, **fit_params)

        if not _can_accelerate(X):
            logger.debug(
                "`StandardScaler.fit_transform` not optimized: "
                "unsupported input"
            )
            return orig_fit_transform(self, X, y, **fit_params)

        logger.debug("`StandardScaler.fit_transform` input data moved to GPU")

        with (
            enable_scipy_array_api(),
            sklearn.config_context(array_api_dispatch=True),
            using_output_type("cupy"),
        ):
            out = orig_fit_transform(self, _to_cupy(X), y, **fit_params)

        _ensure_fitted_on_host(self)
        if GlobalSettings().output_type in (None, "numpy"):
            out = ensure_host(out)

        return out

    cls.fit_transform = fit_transform


def _patch_inverse_transform(cls):
    orig_inverse_transform = cls.inverse_transform

    @functools.wraps(orig_inverse_transform)
    def inverse_transform(self, X, copy=None):
        if not _can_accelerate(X):
            logger.debug(
                "`StandardScaler.inverse_transform` not optimized: "
                "unsupported input"
            )
            return orig_inverse_transform(self, X, copy=copy)

        logger.debug(
            "`StandardScaler.inverse_transform` input data moved to GPU"
        )

        with (
            _fitted_attrs_on_device(self),
            enable_scipy_array_api(),
            sklearn.config_context(array_api_dispatch=True),
            using_output_type("cupy"),
        ):
            out = orig_inverse_transform(self, _to_cupy(X), copy=copy)

        if GlobalSettings().output_type in (None, "numpy"):
            out = ensure_host(out)

        return out

    cls.inverse_transform = inverse_transform


_patch_fit(StandardScaler)
_patch_partial_fit(StandardScaler)
_patch_transform(StandardScaler)
_patch_fit_transform(StandardScaler)
_patch_inverse_transform(StandardScaler)

StandardScaler._cuml_accel_patched = True

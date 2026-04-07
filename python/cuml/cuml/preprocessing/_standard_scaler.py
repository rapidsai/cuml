# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

"""Thin StandardScaler that delegates to scikit-learn with GPU acceleration.

Subclasses :class:`sklearn.preprocessing.StandardScaler` and transparently
moves data to CuPy so that sklearn's array-API dispatch path runs on the GPU.
"""

import contextlib

import cupy as cp
import numpy as np
import scipy.sparse
import sklearn
from sklearn.preprocessing import StandardScaler as SklearnStandardScaler

from cuml.accel._patches.sklearn._utils import enable_scipy_array_api
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


def _to_cupy(X):
    return X if isinstance(X, cp.ndarray) else cp.asarray(X)


def _ensure_host(x):
    return x.get() if isinstance(x, cp.ndarray) else x


def _ensure_fitted_on_host(self):
    """Convert fitted attributes to host arrays when output_type is numpy."""
    if GlobalSettings().output_type in (None, "numpy"):
        for attr in _FITTED_ATTRS:
            val = getattr(self, attr, None)
            if val is not None:
                setattr(self, attr, _ensure_host(val))


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


class StandardScaler(SklearnStandardScaler):
    def fit(self, X, y=None, sample_weight=None):
        if sample_weight is not None or not _can_accelerate(X):
            return super().fit(X, y, sample_weight=sample_weight)

        with (
            enable_scipy_array_api(),
            sklearn.config_context(array_api_dispatch=True),
            using_output_type("cupy"),
        ):
            out = super().fit(_to_cupy(X), y)

        _ensure_fitted_on_host(self)
        return out

    def partial_fit(self, X, y=None, sample_weight=None):
        if sample_weight is not None or not _can_accelerate(X):
            return super().partial_fit(X, y, sample_weight=sample_weight)

        _promote_fitted_to_device(self)
        with (
            enable_scipy_array_api(),
            sklearn.config_context(array_api_dispatch=True),
            using_output_type("cupy"),
        ):
            out = super().partial_fit(_to_cupy(X), y)

        _ensure_fitted_on_host(self)
        return out

    def transform(self, X, copy=None):
        if not _can_accelerate(X):
            return super().transform(X, copy=copy)

        with (
            _fitted_attrs_on_device(self),
            enable_scipy_array_api(),
            sklearn.config_context(array_api_dispatch=True),
            using_output_type("cupy"),
        ):
            out = super().transform(_to_cupy(X), copy=copy)

        if GlobalSettings().output_type in (None, "numpy"):
            out = _ensure_host(out)

        return out

    def fit_transform(self, X, y=None, **fit_params):
        if fit_params.get("sample_weight") is not None or not _can_accelerate(
            X
        ):
            return super().fit_transform(X, y, **fit_params)

        with (
            enable_scipy_array_api(),
            sklearn.config_context(array_api_dispatch=True),
            using_output_type("cupy"),
        ):
            out = super().fit_transform(_to_cupy(X), y, **fit_params)

        _ensure_fitted_on_host(self)
        if GlobalSettings().output_type in (None, "numpy"):
            out = _ensure_host(out)

        return out

    def inverse_transform(self, X, copy=None):
        if not _can_accelerate(X):
            return super().inverse_transform(X, copy=copy)

        with (
            _fitted_attrs_on_device(self),
            enable_scipy_array_api(),
            sklearn.config_context(array_api_dispatch=True),
            using_output_type("cupy"),
        ):
            out = super().inverse_transform(_to_cupy(X), copy=copy)

        if GlobalSettings().output_type in (None, "numpy"):
            out = _ensure_host(out)

        return out

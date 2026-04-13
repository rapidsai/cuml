# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import contextlib
import functools

import cupy as cp
import numpy as np
import scipy.sparse
import sklearn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from cuml.accel._patches.sklearn._utils import enable_scipy_array_api
from cuml.accel.core import logger
from cuml.accel.estimator_proxy import ensure_host
from cuml.internals.global_settings import GlobalSettings
from cuml.internals.outputs import using_output_type

__all__ = ("LabelEncoder", "MinMaxScaler", "StandardScaler")


def _can_accelerate(X, **kwargs):
    """Check if X is suitable for GPU acceleration."""
    if kwargs.get("sample_weight") is not None:
        return False
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


# ---------------------------------------------------------------------------
# Generic fitted-attribute helpers (parameterized by attribute-name tuple)
# ---------------------------------------------------------------------------


def _ensure_fitted_on_host(estimator, fitted_attrs):
    """Convert fitted attributes to host arrays when output_type is numpy."""
    if GlobalSettings().output_type in (None, "numpy"):
        for attr in fitted_attrs:
            val = getattr(estimator, attr, None)
            if val is not None:
                setattr(estimator, attr, ensure_host(val))


def _promote_fitted_to_device(estimator, fitted_attrs):
    """Convert any numpy fitted attributes to cupy in-place.

    After fit, fitted attrs are stored as numpy (for user convenience).
    sklearn's array-API code path needs them in the same namespace as
    the cupy input.
    """
    for attr in fitted_attrs:
        val = getattr(estimator, attr, None)
        if val is not None and isinstance(val, np.ndarray):
            setattr(estimator, attr, cp.asarray(val))


@contextlib.contextmanager
def _fitted_attrs_on_device(estimator, fitted_attrs):
    """Temporarily move fitted attributes to cupy for a computation.

    Use this for read-only methods (transform, inverse_transform) that
    should not permanently alter the stored attributes.  For methods
    that update attributes (partial_fit) call ``_promote_fitted_to_device``
    directly and let ``_ensure_fitted_on_host`` convert back afterward.
    """
    saved = {}
    for attr in fitted_attrs:
        val = getattr(estimator, attr, None)
        if val is not None and isinstance(val, np.ndarray):
            saved[attr] = val
            setattr(estimator, attr, cp.asarray(val))
    try:
        yield
    finally:
        for attr, val in saved.items():
            setattr(estimator, attr, val)


# ---------------------------------------------------------------------------
# Generic method-patch factory
# ---------------------------------------------------------------------------


def _make_method_patch(
    orig_method,
    class_name,
    fitted_attrs,
    *,
    is_fitting=False,
    promotes_fitted=False,
    uses_fitted_ctx=False,
    returns_output=False,
):
    """Return a patched version of *orig_method* that dispatches via CuPy.

    Parameters
    ----------
    orig_method : callable
        The original (unpatched) method.
    class_name : str
        Used in log messages.
    fitted_attrs : tuple[str, ...]
        Names of the estimator's fitted attributes to manage across CPU/GPU.
    is_fitting : bool
        If True, call ``_ensure_fitted_on_host`` after the method runs.
    promotes_fitted : bool
        If True, call ``_promote_fitted_to_device`` before the method runs
        (for incremental methods like ``partial_fit``).
    uses_fitted_ctx : bool
        If True, wrap the call in ``_fitted_attrs_on_device`` so that fitted
        attributes are temporarily on the device (for read-only transform
        methods).
    returns_output : bool
        If True, convert the return value to a host array when appropriate.
    """
    method_name = orig_method.__name__
    log_prefix = f"`{class_name}.{method_name}`"

    @functools.wraps(orig_method)
    def wrapper(self, data, *args, **kwargs):
        if not _can_accelerate(data, **kwargs):
            logger.debug(f"{log_prefix} not optimized: unsupported input")
            return orig_method(self, data, *args, **kwargs)

        logger.debug(f"{log_prefix} input data moved to GPU")

        if promotes_fitted:
            _promote_fitted_to_device(self, fitted_attrs)

        with contextlib.ExitStack() as stack:
            if uses_fitted_ctx:
                stack.enter_context(
                    _fitted_attrs_on_device(self, fitted_attrs)
                )
            stack.enter_context(enable_scipy_array_api())
            stack.enter_context(
                sklearn.config_context(array_api_dispatch=True)
            )
            stack.enter_context(using_output_type("cupy"))
            out = orig_method(self, _to_cupy(data), *args, **kwargs)

        if is_fitting:
            _ensure_fitted_on_host(self, fitted_attrs)

        if returns_output and GlobalSettings().output_type in (None, "numpy"):
            out = ensure_host(out)

        return out

    return wrapper


# ---------------------------------------------------------------------------
# Top-level estimator-patching entry point
# ---------------------------------------------------------------------------


def patch_estimator(cls, fitted_attrs, methods):
    """Monkey-patch *cls* to dispatch its methods through CuPy.

    Parameters
    ----------
    cls : type
        The sklearn estimator class to patch.
    fitted_attrs : tuple[str, ...]
        Names of fitted attributes that need to be migrated between CPU and
        GPU across method calls.
    methods : dict[str, dict]
        Mapping from method name to a dict of keyword arguments forwarded
        to ``_make_method_patch``.
    """
    for method_name, flags in methods.items():
        orig = getattr(cls, method_name)
        patched = _make_method_patch(orig, cls.__name__, fitted_attrs, **flags)
        setattr(cls, method_name, patched)
    cls._cuml_accel_patched = True


# ---------------------------------------------------------------------------
# Per-estimator configuration and registration
# ---------------------------------------------------------------------------

patch_estimator(
    StandardScaler,
    fitted_attrs=("mean_", "var_", "scale_", "n_samples_seen_"),
    methods={
        "fit": dict(is_fitting=True),
        "partial_fit": dict(is_fitting=True, promotes_fitted=True),
        "transform": dict(uses_fitted_ctx=True, returns_output=True),
        "fit_transform": dict(is_fitting=True, returns_output=True),
        "inverse_transform": dict(uses_fitted_ctx=True, returns_output=True),
    },
)

patch_estimator(
    MinMaxScaler,
    fitted_attrs=(
        "scale_",
        "min_",
        "data_min_",
        "data_max_",
        "data_range_",
        "n_samples_seen_",
    ),
    methods={
        "fit": dict(is_fitting=True),
        "partial_fit": dict(is_fitting=True, promotes_fitted=True),
        "transform": dict(uses_fitted_ctx=True, returns_output=True),
        "fit_transform": dict(is_fitting=True, returns_output=True),
        "inverse_transform": dict(uses_fitted_ctx=True, returns_output=True),
    },
)

patch_estimator(
    LabelEncoder,
    fitted_attrs=("classes_",),
    methods={
        "fit": dict(is_fitting=True),
        "fit_transform": dict(is_fitting=True, returns_output=True),
        "transform": dict(uses_fitted_ctx=True, returns_output=True),
        "inverse_transform": dict(uses_fitted_ctx=True, returns_output=True),
    },
)

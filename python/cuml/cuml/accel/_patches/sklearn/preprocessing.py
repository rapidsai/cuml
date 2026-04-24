# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import contextlib
import functools

import cupy as cp
import numpy as np
import scipy.sparse
import sklearn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

from cuml.accel import profilers
from cuml.accel._patches.sklearn._utils import enable_scipy_array_api
from cuml.accel.core import logger
from cuml.accel.estimator_proxy import ensure_host
from cuml.internals.global_settings import GlobalSettings
from cuml.internals.outputs import using_output_type

__all__ = ("LabelEncoder", "MinMaxScaler", "StandardScaler")


def _is_accelerable_dtype(arr):
    """True if `arr` has a numeric dtype that the cupy+array_api path can handle.

    Excludes object, character, complex, and float16 (float16 is not reliably
    supported by sklearn's array_api dispatch path).
    """
    if not hasattr(arr, "dtype"):
        return False
    dt = arr.dtype
    if dt == np.object_ or dt == np.float16:
        return False
    if np.issubdtype(dt, np.complexfloating):
        return False
    if np.issubdtype(dt, np.character):
        return False
    return np.issubdtype(dt, np.floating) or np.issubdtype(dt, np.integer)


def _to_cupy(X):
    return X if isinstance(X, cp.ndarray) else cp.asarray(X)


def _can_accelerate_standard_scaler(estimator, X, **kwargs):
    # sample_weight is accepted by (partial_)fit. The cupy+array_api path for
    # _incremental_mean_and_var has not been validated with cupy sample weights,
    # so fall back to CPU when one is supplied.
    if kwargs.get("sample_weight") is not None:
        return False
    if hasattr(X, "columns"):
        return False
    # sklearn accepts sparse input when with_mean=False, but the cupy+array_api
    # path does not support scipy sparse matrices.
    if scipy.sparse.issparse(X):
        return False
    return _is_accelerable_dtype(X)


def _can_accelerate_min_max_scaler(estimator, X, **kwargs):
    if hasattr(X, "columns"):
        return False
    # sklearn raises TypeError for sparse input; nothing to accelerate there.
    if scipy.sparse.issparse(X):
        return False
    return _is_accelerable_dtype(X)


def _can_accelerate_label_encoder(estimator, y, **kwargs):
    # LabelEncoder operates on 1D y which may contain strings on CPU.
    # cupy cannot hold string arrays, so only numeric y qualifies.
    if hasattr(y, "columns"):
        return False
    if scipy.sparse.issparse(y):
        return False
    if not _is_accelerable_dtype(y):
        return False
    # cupy cannot hold string arrays, so if classes_ has a non-numeric dtype
    # (e.g. the encoder was fitted on strings), any accelerated call will fail:
    # transform() tries to cast y to classes_.dtype, and inverse_transform()
    # calls xp.take(classes_, y) where xp=cp cannot handle a numpy string array.
    classes = getattr(estimator, "classes_", None)
    if classes is not None and not np.issubdtype(classes.dtype, np.number):
        return False
    return True


def _has_non_default_output_container(estimator):
    """Return True when sklearn output wrapping is configured."""
    output_config = getattr(estimator, "_sklearn_output_config", {})
    transform_output = output_config.get("transform")
    if transform_output is not None:
        return transform_output != "default"
    return sklearn.get_config().get("transform_output", "default") != "default"


# ---------------------------------------------------------------------------
# Generic fitted-attribute helpers
# ---------------------------------------------------------------------------


def _get_fitted_attrs(estimator):
    """Discover fitted attributes by sklearn's trailing-underscore convention."""
    return [
        attr
        for attr in vars(estimator)
        if attr.endswith("_") and not attr.startswith("_")
    ]


def _ensure_fitted_on_host(estimator):
    """Convert fitted attributes to host (numpy) arrays.

    Fitted attrs are stored as numpy regardless of the current
    ``GlobalSettings().output_type`` so that the unaccelerated fallback
    path (e.g. when a non-default ``transform_output`` is configured) can
    operate on them without mixing cupy fitted attrs with a numpy input.
    Accelerated transform paths temporarily promote these back to the
    device via :func:`_fitted_attrs_on_device`.
    """
    for attr in _get_fitted_attrs(estimator):
        val = getattr(estimator, attr, None)
        if val is not None:
            setattr(estimator, attr, ensure_host(val))


def _promote_fitted_to_device(estimator):
    """Convert any numpy fitted attributes to cupy in-place.

    After fit, fitted attrs are stored as numpy (for user convenience).
    sklearn's array-API code path needs them in the same namespace as
    the cupy input.
    """
    for attr in _get_fitted_attrs(estimator):
        val = getattr(estimator, attr, None)
        if (
            val is not None
            and isinstance(val, np.ndarray)
            and np.issubdtype(val.dtype, np.number)
        ):
            setattr(estimator, attr, cp.asarray(val))


@contextlib.contextmanager
def _fitted_attrs_on_device(estimator):
    """Temporarily move fitted attributes to cupy for a computation.

    Use this for read-only methods (transform, inverse_transform) that
    should not permanently alter the stored attributes.  For methods
    that update attributes (partial_fit) call ``_promote_fitted_to_device``
    directly and let ``_ensure_fitted_on_host`` convert back afterward.
    """
    saved = {}
    for attr in _get_fitted_attrs(estimator):
        val = getattr(estimator, attr, None)
        if (
            val is not None
            and isinstance(val, np.ndarray)
            and np.issubdtype(val.dtype, np.number)
        ):
            saved[attr] = val
            setattr(estimator, attr, cp.asarray(val))
    try:
        yield
    finally:
        for attr, val in saved.items():
            setattr(estimator, attr, val)


# ---------------------------------------------------------------------------
# Method-flag inference
# ---------------------------------------------------------------------------

_TRANSFORMER_METHODS = (
    "fit",
    "partial_fit",
    "transform",
    "fit_transform",
    "inverse_transform",
)


def _infer_method_flags(method_name):
    """Derive _make_method_patch flags from a method name.

    Covers the standard transformer method vocabulary:
    fit, partial_fit, transform, fit_transform, inverse_transform.
    """
    is_fit = "fit" in method_name
    is_transform = "transform" in method_name
    return dict(
        is_fitting=is_fit,
        promotes_fitted=(method_name == "partial_fit"),
        uses_fitted_ctx=(is_transform and not is_fit),
        returns_output=is_transform,
    )


# ---------------------------------------------------------------------------
# Generic method-patch factory
# ---------------------------------------------------------------------------


def _make_method_patch(
    orig_method,
    class_name,
    *,
    can_accelerate,
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
    can_accelerate : callable(estimator, data, **kwargs) -> bool
        Per-estimator predicate; when False the original method is called.
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
    qualname = f"{class_name}.{method_name}"

    @functools.wraps(orig_method)
    def wrapper(self, data, *args, **kwargs):
        if returns_output and _has_non_default_output_container(self):
            logger.debug(
                f"{log_prefix} not optimized: non-default output container"
            )
            with profilers.track_cpu_call(
                qualname, reason="non-default output container"
            ):
                return orig_method(self, data, *args, **kwargs)

        if not can_accelerate(self, data, **kwargs):
            logger.debug(f"{log_prefix} not optimized: unsupported input")
            with profilers.track_cpu_call(
                qualname, reason="unsupported input"
            ):
                return orig_method(self, data, *args, **kwargs)

        logger.debug(f"{log_prefix} input data moved to GPU")

        with profilers.track_gpu_call(qualname):
            if promotes_fitted:
                _promote_fitted_to_device(self)

            with contextlib.ExitStack() as stack:
                if uses_fitted_ctx:
                    stack.enter_context(_fitted_attrs_on_device(self))
                stack.enter_context(enable_scipy_array_api())
                stack.enter_context(
                    sklearn.config_context(array_api_dispatch=True)
                )
                stack.enter_context(using_output_type("cupy"))
                out = orig_method(self, _to_cupy(data), *args, **kwargs)

            if is_fitting:
                _ensure_fitted_on_host(self)

            if returns_output and GlobalSettings().output_type in (
                None,
                "numpy",
            ):
                out = ensure_host(out)

        return out

    return wrapper


# ---------------------------------------------------------------------------
# Top-level estimator-patching entry point
# ---------------------------------------------------------------------------


def patch_estimator(cls, can_accelerate, methods=None):
    """Monkey-patch *cls* to dispatch its methods through CuPy.

    Parameters
    ----------
    cls : type
        The sklearn estimator class to patch.
    can_accelerate : callable(estimator, data, **kwargs) -> bool
        Per-estimator predicate used by every patched method to decide
        whether to route through the cupy+array_api path.
    methods : sequence of str or None
        Names of methods to patch.  When *None* (default), all methods in
        ``_TRANSFORMER_METHODS`` that exist on *cls* are patched
        automatically, with flags inferred by ``_infer_method_flags``.
    """
    if methods is None:
        methods = [m for m in _TRANSFORMER_METHODS if hasattr(cls, m)]
    for method_name in methods:
        flags = _infer_method_flags(method_name)
        orig = getattr(cls, method_name)
        patched = _make_method_patch(
            orig, cls.__name__, can_accelerate=can_accelerate, **flags
        )
        setattr(cls, method_name, patched)
    cls._cuml_accel_patched = True


# ---------------------------------------------------------------------------
# Per-estimator configuration and registration
# ---------------------------------------------------------------------------

patch_estimator(StandardScaler, _can_accelerate_standard_scaler)
patch_estimator(MinMaxScaler, _can_accelerate_min_max_scaler)
patch_estimator(LabelEncoder, _can_accelerate_label_encoder)

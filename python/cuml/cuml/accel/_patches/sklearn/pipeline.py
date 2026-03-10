# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import functools
import inspect

import cupy as cp
from cupyx.scipy.sparse import issparse as is_cp_sparse
from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import available_if

from cuml.accel.estimator_proxy import is_proxy
from cuml.internals.global_settings import GlobalSettings
from cuml.internals.outputs import using_output_type

__all__ = ("Pipeline",)


def get_output_type(pipeline, reverse=False):
    """Determine the output type to use for a pipeline method.

    If all cupy-producing steps are only followed by cupy-consuming steps after
    them, then we may enable cupy outputs. Otherwise we need to fallback to
    numpy. A pipeline like ``numpy -> numpy -> cupy -> cupy`` is fine to enable
    cupy for, but ``numpy -> cupy -> numpy -> cupy`` isn't.
    """

    def flat_steps(pipeline):
        """Iterate over steps potentially nested pipelines"""
        for name, step in (
            reversed(pipeline.steps) if reverse else pipeline.steps
        ):
            if step in (None, "passthrough"):
                continue
            if isinstance(step, Pipeline):
                yield from flat_steps(step)
            else:
                yield step

    step_iter = flat_steps(pipeline)

    # Skip over any non-cupy producing steps
    for step in step_iter:
        if is_proxy(step):
            break
    else:
        # No steps produce cupy
        return "numpy"

    # If any remaining steps don't consume cupy, we need to fallback to numpy
    for step in step_iter:
        if not is_proxy(step):
            return "numpy"
    # Tail of the pipeline supports cupy, we can use cupy
    return "cupy"


def patch_method(name):
    """Patch a sklearn Pipeline method to reduce device<->host transfers."""
    orig_method = inspect.getattr_static(Pipeline, name)
    # Unwrap @available_if decorated methods
    if (check := getattr(orig_method, "check", None)) is not None:
        orig_method = orig_method.fn

    # `inverse_transform` processes steps in reverse
    reverse = name == "inverse_transform"

    @functools.wraps(orig_method)
    def method(self, *args, **kwargs):
        if name.startswith("fit"):
            # Validate hyperparameters first if a fit call
            self._validate_params()

        # Run the original method within the proper output type context
        with using_output_type(get_output_type(self, reverse=reverse)):
            out = orig_method(self, *args, **kwargs)

        # Transform output to numpy/scipy.sparse if requested
        if GlobalSettings().output_type in (None, "numpy") and (
            isinstance(out, cp.ndarray) or is_cp_sparse(out)
        ):
            out = out.get()

        return out

    # Rewrap @available_if decorated methods
    if check is not None:
        method = available_if(check)(method)

    setattr(Pipeline, name, method)


# These methods run pipeline operations in a series, and need to be
# patched to reduce data movement between a series of accelerated
# estimators.
for method_name in [
    "decision_function",
    "fit",
    "fit_predict",
    "fit_transform",
    "inverse_transform",
    "predict",
    "predict_log_proba",
    "predict_proba",
    "score",
    "score_samples",
    "transform",
]:
    patch_method(method_name)

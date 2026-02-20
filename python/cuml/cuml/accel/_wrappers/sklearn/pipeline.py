#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""Thin Pipeline wrapper for cuml.accel that keeps intermediate data on device."""

from __future__ import annotations

from dataclasses import replace

from sklearn.pipeline import Pipeline as _SklearnPipeline
from sklearn.utils._tags import TransformerTags

from cuml._thirdparty.sklearn.preprocessing._pipeline import (
    Pipeline as _AccelPipeline,
)
from cuml.accel.estimator_proxy import ProxyBase

__all__ = ("Pipeline", "make_pipeline")


class Pipeline(ProxyBase):
    """cuml.accel proxy for sklearn.pipeline.Pipeline.

    Keeps intermediate data on device when possible. Subclasses ProxyBase like
    other accel estimators; the device-aware implementation is _AccelPipeline.
    """

    _gpu_class = _AccelPipeline

    # _AccelPipeline is a sklearn Pipeline subclass, not a cuML estimator;
    # it has no _get_tags()["X_types_gpu"]. Override so ProxyBase does not expect it.
    _gpu_supports_sparse = False
    # Access to steps/named_steps must sync fitted state from _gpu so step estimators
    # (and their fitted attributes) are available on _cpu.
    _other_attributes = frozenset({"steps", "named_steps"})

    def __sklearn_tags__(self):
        # sklearn's Pipeline.__sklearn_tags__() can leave transformer_tags=None
        # when the last step is not a transformer or steps are not yet set.
        # Common tests require transformer_tags to be set for any estimator
        # with transform(). Override __sklearn_tags__ (not just _get_tags)
        # because get_tags() calls __sklearn_tags__, not _get_tags.
        tags = self._cpu.__sklearn_tags__()
        if tags.transformer_tags is None and hasattr(self._cpu, "transform"):
            return replace(tags, transformer_tags=TransformerTags())
        return tags

    def __len__(self):
        """Return the number of steps in the pipeline (delegate to CPU; step count needs no sync)."""
        return len(self._cpu)

    def __getitem__(self, ind):
        """Return a sub-pipeline or a single estimator; delegate to CPU and wrap slices."""
        self._sync_attrs_to_cpu()
        result = self._cpu[ind]
        if isinstance(result, _SklearnPipeline):
            return type(self)._reconstruct_from_cpu(result)
        return result


def make_pipeline(*steps, memory=None, transform_input=None, verbose=False):
    """Construct a Pipeline from the given estimators (cuml.accel version).

    Delegates to sklearn's _name_estimators then builds our Pipeline so that
    make_pipeline(...) returns an accelerated Pipeline instance.
    """
    from sklearn.pipeline import _name_estimators

    return Pipeline(
        _name_estimators(steps),
        memory=memory,
        transform_input=transform_input,
        verbose=verbose,
    )

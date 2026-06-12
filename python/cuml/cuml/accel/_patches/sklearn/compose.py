# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import functools
import inspect

from sklearn.compose import ColumnTransformer
from sklearn.utils.metaestimators import available_if

from cuml.internals.outputs import using_output_type

__all__ = ("ColumnTransformer",)


def patch_method(name):
    """Patch a ColumnTransformer method to keep sklearn composition on host."""
    orig_method = inspect.getattr_static(ColumnTransformer, name)
    if (check := getattr(orig_method, "check", None)) is not None:
        orig_method = orig_method.fn

    @functools.wraps(orig_method)
    def method(self, *args, **kwargs):
        with using_output_type("numpy"):
            return orig_method(self, *args, **kwargs)

    if check is not None:
        method = available_if(check)(method)

    setattr(ColumnTransformer, name, method)


for method_name in ["fit", "fit_transform", "transform"]:
    patch_method(method_name)

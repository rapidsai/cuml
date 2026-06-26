# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import functools

from sklearn.compose import ColumnTransformer

from cuml.internals.outputs import using_output_type

__all__ = ("ColumnTransformer",)


def patch_method(name):
    """Patch a ColumnTransformer method to ensure results returned as numpy."""
    orig_method = getattr(ColumnTransformer, name)

    @functools.wraps(orig_method)
    def method(self, *args, **kwargs):
        with using_output_type("numpy"):
            return orig_method(self, *args, **kwargs)

    setattr(ColumnTransformer, name, method)


for method_name in ["fit", "fit_transform", "transform"]:
    patch_method(method_name)

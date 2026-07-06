#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
__all__ = ("Lars",)


def __getattr__(name):
    if name == "Lars":
        import warnings

        from cuml.linear_model.lars import Lars

        warnings.warn(
            "`cuml.experimental.linear_model.Lars` was deprecated in 26.08 "
            "and will be removed in 26.10. Please use `cuml.linear_model.Lars` "
            "instead.",
            FutureWarning,
        )
        return Lars
    raise AttributeError(f"module {__name__} has no attribute {name}")

#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
__all__ = ("NotFittedError",)  # noqa


def __getattr__(name):
    if name == "NotFittedError":
        import warnings

        from sklearn.exceptions import NotFittedError

        warnings.warn(
            "`cuml.common.exceptions.NotFittedError` was deprecated in 26.04 "
            "and will be removed in 26.06. Please use "
            "`sklearn.exceptions.NotFittedError` instead.",
            FutureWarning,
            stacklevel=2,
        )
        return NotFittedError
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")

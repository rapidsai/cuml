# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Shared utilities for sklearn patch modules."""

import contextlib
import os

import scipy._lib._array_api as _scipy_array_api


@contextlib.contextmanager
def enable_scipy_array_api():
    """Enable scipy's array API support for the duration of a block.

    Sets the SCIPY_ARRAY_API env var (checked by sklearn's config validation)
    and updates scipy's cached config (in case scipy had already been
    imported). Both are restored on exit.

    This is required before entering
    ``sklearn.config_context(array_api_dispatch=True)`` so that sklearn does
    not raise a RuntimeError about scipy's array API support being absent.
    """
    old_env = os.environ.get("SCIPY_ARRAY_API")
    os.environ["SCIPY_ARRAY_API"] = "1"

    old_cached = _scipy_array_api._GLOBAL_CONFIG["SCIPY_ARRAY_API"]
    _scipy_array_api._GLOBAL_CONFIG["SCIPY_ARRAY_API"] = "1"

    try:
        yield
    finally:
        if old_env is None:
            os.environ.pop("SCIPY_ARRAY_API", None)
        else:
            os.environ["SCIPY_ARRAY_API"] = old_env

        _scipy_array_api._GLOBAL_CONFIG["SCIPY_ARRAY_API"] = old_cached

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import functools
import warnings

from cuml.internals.outputs import in_internal_context


def warn_deprecated_tsa_api(api_name, *, stacklevel=2):
    warnings.warn(
        (
            f"`{api_name}`, along with the entire `cuml.tsa` module, was "
            "deprecated in version 26.08 and will be removed in version 26.12."
        ),
        FutureWarning,
        stacklevel=stacklevel,
    )


def deprecated_tsa_api(api_name):
    def decorator(func):
        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            if not in_internal_context():
                warn_deprecated_tsa_api(api_name, stacklevel=3)
            return func(*args, **kwargs)

        return wrapped

    return decorator

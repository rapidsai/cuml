# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import functools

from sklearn.utils._array_api import (
    _check_array_api_dispatch as _orig_check_array_api_dispatch,
)

from cuml.internals.outputs import in_internal_context

__all__ = ("_check_array_api_dispatch",)


@functools.wraps(_orig_check_array_api_dispatch)
def _check_array_api_dispatch(array_api_dispatch):
    # sklearn's array-api support requires setting SCIPY_ARRAY_API=1, even
    # though all uses we need it for don't rely on scipy. To work around this,
    # we patch sklearn to disable the check when running within a cuml
    # estimator. Usage outside of cuml estimators will still result in the
    # proper error.
    if not in_internal_context():
        _orig_check_array_api_dispatch(array_api_dispatch)

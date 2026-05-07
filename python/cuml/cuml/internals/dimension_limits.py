#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""Guards for Python bindings that pass array dimensions to C/CUDA as 32-bit types."""

from __future__ import annotations

import numpy as np

# Signed 32-bit int maximum (typical C ``int`` on cuML-supported platforms).
INT32_MAX = 2_147_483_647
INT32_MIN = -INT32_MAX - 1
UINT32_MAX = 4_294_967_295


def values_fit_int32(**values: int) -> None:
    """
    Verify each scalar fits a C signed ``int``; raise ``ValueError`` if not.

    Unlike ``dims_within_int_limits``, negative values are allowed (e.g. label
    ranges passed to legacy kernels as ``int``).
    """
    for name, value in values.items():
        v = int(value)
        if v < INT32_MIN or v > INT32_MAX:
            raise ValueError(
                f"{name}={value!r} is outside the range representable as a "
                f"32-bit signed integer [{INT32_MIN}, {INT32_MAX}]; the binding "
                "would truncate when passing this value to native code."
            )


def dims_within_int_limits(**dims: int) -> None:
    """
    Verify dimensions fit a C ``int``; raise ``ValueError`` if any do not.

    Several legacy CUDA entry points take row counts, column counts, or lengths
    as C ``int``. Values outside ``[0, INT32_MAX]`` are silently truncated in
    Cython, which corrupts kernel launches.
    """
    for name, value in dims.items():
        v = int(value)
        if v < 0:
            raise ValueError(f"{name} must be non-negative, got {value!r}")
        if v > INT32_MAX:
            raise ValueError(
                f"{name}={value!r} exceeds the maximum value supported by this "
                f"binding (<= {INT32_MAX}); larger inputs would be truncated when "
                "passed to native code as a 32-bit signed integer."
            )


def dims_within_uint32_limits(**dims: int) -> None:
    """Verify dimensions fit ``uint32_t``; raise ``ValueError`` if any do not."""
    for name, value in dims.items():
        v = int(value)
        if v < 0:
            raise ValueError(f"{name} must be non-negative, got {value!r}")
        if v > UINT32_MAX:
            raise ValueError(
                f"{name}={value!r} exceeds uint32_t maximum ({UINT32_MAX})."
            )


def dims_within_size_t_limits(**dims: int) -> None:
    """
    Verify values fit ``size_t`` on this platform; raise ``ValueError`` if not.

    Uses ``numpy.uintp`` range (same width as ``size_t`` / ``uintptr_t`` on
    supported builds) for the upper bound.
    """
    maxv = int(np.iinfo(np.uintp).max)
    for name, value in dims.items():
        v = int(value)
        if v < 0:
            raise ValueError(f"{name} must be non-negative, got {value!r}")
        if v > maxv:
            raise ValueError(
                f"{name}={value!r} exceeds the maximum value representable as "
                f"size_t on this platform ({maxv})."
            )

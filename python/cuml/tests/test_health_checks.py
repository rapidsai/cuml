# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import inspect

import pytest

from cuml.health_checks import _checks
from cuml.health_checks.__main__ import _CHECKS


def _get_public_check_functions():
    """Return all public functions defined in _checks module."""
    return {
        name: obj
        for name, obj in inspect.getmembers(_checks, inspect.isfunction)
        if not name.startswith("_") and obj.__module__ == _checks.__name__
    }


_CHECK_IDS = [name for name, _ in _CHECKS]


@pytest.mark.parametrize("name,check_fn", _CHECKS, ids=_CHECK_IDS)
def test_health_check(name, check_fn):
    """Each registered health check should pass."""
    check_fn(verbose=True)


def test_all_checks_registered():
    """Every public function in _checks must appear in _CHECKS."""
    registered_fns = {fn for _, fn in _CHECKS}
    public_fns = _get_public_check_functions()
    missing = {
        name for name, fn in public_fns.items() if fn not in registered_fns
    }
    assert not missing, (
        f"Public check functions not registered in _CHECKS: {missing}"
    )


def test_check_function_signatures():
    """All check functions must accept (verbose, **kwargs) per the rapids doctor contract."""
    for name, check_fn in _CHECKS:
        sig = inspect.signature(check_fn)
        params = list(sig.parameters.values())

        assert len(params) >= 2, (
            f"{name}: expected at least 2 parameters (verbose, **kwargs), "
            f"got {len(params)}"
        )
        assert params[0].name == "verbose", (
            f"{name}: first parameter should be 'verbose', "
            f"got '{params[0].name}'"
        )
        assert params[0].default is False, (
            f"{name}: 'verbose' should default to False, "
            f"got {params[0].default!r}"
        )
        assert params[-1].kind == inspect.Parameter.VAR_KEYWORD, (
            f"{name}: last parameter should be **kwargs, got {params[-1]}"
        )

#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

"""Run cuML health checks when invoked as ``python -m cuml.health_checks``."""

import argparse
import sys

from cuml.health_checks import (
    accel_basic_check,
    accel_cli_check,
    functional_check,
    import_check,
)

_CHECKS = [
    ("import", import_check),
    ("functional", functional_check),
    ("accel-basic", accel_basic_check),
    ("accel-cli", accel_cli_check),
]


_CHECK_NAMES = [name for name, _ in _CHECKS]


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m cuml.health_checks",
        description="Run cuML health checks.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Print extra output when a check passes.",
    )
    parser.add_argument(
        "checks",
        nargs="*",
        metavar="CHECK",
        choices=_CHECK_NAMES,
        help=(
            f"Names of checks to run (default: all). "
            f"Available: {', '.join(_CHECK_NAMES)}"
        ),
    )
    args = parser.parse_args(argv)

    selected = set(args.checks) if args.checks else None
    failed = False
    for name, check_fn in _CHECKS:
        if selected is not None and name not in selected:
            continue
        try:
            result = check_fn(verbose=args.verbose)
            print(f"{name}: OK")
            if args.verbose and result:
                print(f"  {result}")
        except Exception as e:
            print(f"{name}: FAIL - {e}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

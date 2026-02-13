#
# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""Shared CLI interface for the benchmark runner.

Supports both full mode (python -m cuml.benchmark) and standalone mode
(python run_benchmark.py from the benchmark directory).
"""

import sys

try:
    from cuml.benchmark.core import build_parser, run_benchmark
except ImportError:
    from core import build_parser, run_benchmark


def main(argv=None):
    """Parse arguments and run the benchmark. Returns exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

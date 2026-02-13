#
# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""Core benchmark entry point with dual-import support.

Allows the same code to be used from the installed package (cuml.benchmark)
or from the standalone benchmark directory without cuML installed.
"""

try:
    from cuml.benchmark.run_benchmarks import build_parser, run_benchmark
except ImportError:
    from run_benchmarks import build_parser, run_benchmark

__all__ = ["build_parser", "run_benchmark"]

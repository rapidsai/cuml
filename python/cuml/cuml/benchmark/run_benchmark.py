#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""Standalone entry point for running benchmarks without cuML installed.

Usage from the benchmark directory:
  python run_benchmark.py [options] [algorithms...]

This script adds the benchmark directory to sys.path so that core, cli,
and other benchmark modules can be imported without installing cuML.
"""

import os
import sys

# Allow direct execution from the benchmark directory (standalone mode)
_benchmark_dir = os.path.dirname(os.path.abspath(__file__))
if _benchmark_dir not in sys.path:
    sys.path.insert(0, _benchmark_dir)

from cli import main

sys.exit(main())

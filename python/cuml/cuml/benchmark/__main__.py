#
# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""Entry point for python -m cuml.benchmark (full mode, requires cuML)."""

import sys

from cuml.benchmark.cli import main

if __name__ == "__main__":
    sys.exit(main())

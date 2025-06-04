#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

set -euo pipefail

# Support customizing the ctests' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/libcuml/"

ctest --output-on-failure --no-tests=error "$@"

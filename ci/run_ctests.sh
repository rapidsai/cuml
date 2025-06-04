#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -euo pipefail

source ./ci/use_conda_packages_from_prs.sh


# Support customizing the ctests' install location
cd "${INSTALL_PREFIX:-${CONDA_PREFIX:-/usr}}/bin/gtests/libcuml/"

ctest --output-on-failure --no-tests=error "$@"

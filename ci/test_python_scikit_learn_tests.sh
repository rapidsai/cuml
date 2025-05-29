#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

# Support invoking test script outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../ || exit 1

# Common setup steps shared by Python test jobs
source ./ci/test_python_common.sh

python -c "import cuml"

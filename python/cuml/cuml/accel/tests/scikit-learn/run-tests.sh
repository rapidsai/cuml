#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

# This script runs scikit-learn tests with the cuml.accel plugin.
# Any arguments passed to this script are forwarded directly to pytest.
#
# Example usage:
#   ./run-tests.sh                     # Run all tests
#   ./run-tests.sh -v -k test_kmeans   # Run specific test with verbosity
#   ./run-tests.sh -x --pdb            # Stop on first failure and debug

set -eu

# Get CUDA version
CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)

# Base arguments
PYTEST_ARGS="-p cuml.accel --pyargs sklearn -v --xfail-list=\"$(dirname "$0")/xfail-list.yaml\""

# Skip sequential tests for CUDA 11.x until
# https://github.com/rapidsai/cuml/issues/6622 is resolved
if [[ "$CUDA_VERSION" == "11"* ]]; then
    PYTEST_ARGS="$PYTEST_ARGS -k \"not test_sequential\""
fi

# Run pytest with all arguments
eval "pytest $PYTEST_ARGS" "$@"

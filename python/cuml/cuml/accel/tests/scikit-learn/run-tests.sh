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
PYTEST_ARGS="-p cuml.accel --pyargs sklearn --xfail-list=\"$(dirname "$0")/xfail-list.yaml\""

PYTEST_ARGS="$PYTEST_ARGS -k \"not test_sequential\""

# Run pytest with all arguments
eval "pytest $PYTEST_ARGS" "$@"

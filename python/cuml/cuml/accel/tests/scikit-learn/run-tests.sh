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

# Base arguments
PYTEST_ARGS=("-p" "cuml.accel" "--pyargs" "sklearn" "--xfail-list=$(dirname "$0")/xfail-list.yaml")
# Fail on unmatched xfail tests
PYTEST_ARGS+=("-W" "error::cuml.accel.pytest_plugin.UnmatchedXfailTests")

# Run pytest with all arguments
pytest "${PYTEST_ARGS[@]}" "$@"

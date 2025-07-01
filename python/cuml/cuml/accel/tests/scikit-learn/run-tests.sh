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

file=$CONDA_PREFIX/lib/python3.13/site-packages/sklearn/utils/discovery.py
if ! grep -qF 'estimators = {name: est for name, est in estimators}' "$file"; then
  sed -i "/return sorted(set(estimators), key=itemgetter(0))/i\\
    estimators = {name: est for name, est in estimators}\\
    estimators = [(name, est) for name, est in estimators.items()]" "$file"
fi

# Base arguments
PYTEST_ARGS=("-p" "cuml.accel" "--pyargs" "sklearn" "--xfail-list=$(dirname "$0")/xfail-list.yaml")
# Fail on unmatched xfail tests
PYTEST_ARGS+=("-W" "error::cuml.accel.pytest_plugin.UnmatchedXfailTests")

# Run pytest with all arguments
pytest "${PYTEST_ARGS[@]}" "$@"

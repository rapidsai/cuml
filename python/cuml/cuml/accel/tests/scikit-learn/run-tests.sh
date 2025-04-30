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

cd "$(dirname "$0")"
export PYTHONPATH=$PWD
pytest -p cuml.accel --pyargs sklearn -v -s \
    --xfail-list="$(dirname "$0")/xfail-list.yaml" \
    "$@"

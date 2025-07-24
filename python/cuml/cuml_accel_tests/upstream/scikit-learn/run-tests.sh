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

THIS_DIRECTORY=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

# Run the sklearn test suite
pytest -p cuml.accel \
    --pyargs sklearn \
    --rootdir="${THIS_DIRECTORY}" \
    --config-file="${THIS_DIRECTORY}/../pytest.ini" \
    --xfail-list="${THIS_DIRECTORY}/xfail-list.yaml" \
    "$@"

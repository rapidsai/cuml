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

UMAP_TAG="release-0.5.7"

# cd into this directory so we can use relative paths
THIS_DIRECTORY=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$THIS_DIRECTORY"

# Shallow clone the tag if not already cloned
if [ ! -d umap-upstream ]; then
    git clone --branch $UMAP_TAG --depth 1 "https://github.com/lmcinnes/umap.git" umap-upstream
fi

# Run upstream tests
pytest -p cuml.accel umap-upstream/umap/tests/ \
    --xfail-list=xfail-list.yaml \
    -W "error::cuml.accel.pytest_plugin.UnmatchedXfailTests" \
    "$@"

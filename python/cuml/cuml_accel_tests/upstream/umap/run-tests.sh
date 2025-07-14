#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

# This script runs the umap tests with the cuml.accel plugin.
# Any arguments passed to this script are forwarded directly to pytest.
#
# Example usage:
#   ./run-tests.sh                     # Run all tests
#   ./run-tests.sh -v -k test_foo      # Run specific test with verbosity
#   ./run-tests.sh -x --pdb            # Stop on first failure and debug

set -eu

UMAP_TAG="release-0.5.7"

THIS_DIRECTORY=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
UMAP_REPO="${THIS_DIRECTORY}/umap-upstream"

# Shallow clone the tag if not already cloned
if [ ! -d "$UMAP_REPO" ]; then
    git clone --branch $UMAP_TAG --depth 1 "https://github.com/lmcinnes/umap.git" "$UMAP_REPO"
fi

# Run upstream tests
pytest -p cuml.accel \
    "${UMAP_REPO}/umap/tests/" \
    --rootdir="${THIS_DIRECTORY}" \
    --config-file="${THIS_DIRECTORY}/../pytest.ini" \
    --xfail-list="${THIS_DIRECTORY}/xfail-list.yaml" \
    "$@"

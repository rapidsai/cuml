#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# This script runs the umap tests with the cuml.accel plugin.
# Any arguments passed to this script are forwarded directly to pytest.
#
# Example usage:
#   ./run-tests.sh                     # Run all tests
#   ./run-tests.sh -v -k test_foo      # Run specific test with verbosity
#   ./run-tests.sh -x --pdb            # Stop on first failure and debug

set -eu

UMAP_VERSION=$(python -c "import umap; print(umap.__version__)")
UMAP_TAG="release-${UMAP_VERSION}"

THIS_DIRECTORY=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
UMAP_REPO="${THIS_DIRECTORY}/umap-upstream"

# Clone if not already present, then check out the matching tag
if [ ! -d "$UMAP_REPO" ]; then
    git clone "https://github.com/lmcinnes/umap.git" "$UMAP_REPO"
fi
git -C "$UMAP_REPO" fetch --tags
git -C "$UMAP_REPO" checkout "$UMAP_TAG"

# Run upstream tests
pytest -p cuml.accel \
    "${UMAP_REPO}/umap/tests/" \
    --rootdir="${UMAP_REPO}" \
    --config-file="${THIS_DIRECTORY}/../pytest.ini" \
    --xfail-list="${THIS_DIRECTORY}/xfail-list.yaml" \
    "$@"

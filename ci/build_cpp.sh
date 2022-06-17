#!/bin/bash
set -e

# Update env vars
source rapids-env-update

# Check environment
source ci/check_environment.sh

# Use Ninja to build, setup Conda Build Dir
export CMAKE_GENERATOR="Ninja"
# ucx-py version
export UCX_PY_VERSION='0.27.*'

################################################################################
# BUILD - Conda package builds (LIBCUGRAPH)
################################################################################
gpuci_logger "Begin cpp build"

gpuci_mamba_retry mambabuild \
  --croot /tmp/conda-bld-workspace \
  --output-folder /tmp/conda-bld-output \
  conda/recipes/libcuml

rapids-upload-conda-to-s3 cpp

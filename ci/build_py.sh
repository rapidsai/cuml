#!/bin/bash
set -e

# Update env vars
source rapids-env-update

# Check environment
source ci/check_environment.sh

# ucx-py version
export UCX_PY_VERSION='0.27.*'

################################################################################
# BUILD - Conda package builds (CUGRAPH)
################################################################################
gpuci_logger "Begin py build"

# Python Build Stage
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

gpuci_mamba_retry mambabuild \
  -c "${CPP_CHANNEL}" \
  --croot /tmp/conda-bld-workspace \
  --output-folder /tmp/conda-bld-output \
  conda/recipes/cuml

rapids-upload-conda-to-s3 python

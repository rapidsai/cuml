#!/bin/bash
# Copyright (c) 2018-2021, NVIDIA CORPORATION.

set -e

# Setup 'gpuci_retry' for upload retries (results in 4 total attempts)
export GPUCI_RETRY_MAX=3
export GPUCI_RETRY_SLEEP=30

# Set default label options if they are not defined elsewhere
export LABEL_OPTION=${LABEL_OPTION:-"--label main"}

# Skip uploads unless BUILD_MODE == "branch"
if [[ ${BUILD_MODE} != "branch" ]]; then
  echo "Skipping upload"
  return 0
fi

# Skip uploads if there is no upload key
if [[ -z "$MY_UPLOAD_KEY" ]]; then
  echo "No upload key"
  return 0
fi

################################################################################
# SETUP - Get conda file output locations
################################################################################

gpuci_logger "Get conda file output locations"

export LIBCUML_FILE=`conda build --no-build-id --croot ${CONDA_BLD_DIR} conda/recipes/libcuml --output`
export CUML_FILE=`conda build --croot ${CONDA_BLD_DIR} conda/recipes/cuml --python=$PYTHON --output`

################################################################################
# UPLOAD - Conda packages
################################################################################

gpuci_logger "Starting conda uploads"

if [[ "$BUILD_LIBCUML" == "1" && "$UPLOAD_LIBCUML" == "1" ]]; then
  test -e ${LIBCUML_FILE}
  echo "Upload libcuml"
  echo ${LIBCUML_FILE}
  gpuci_retry anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing ${LIBCUML_FILE}
fi

if [[ "$BUILD_CUML" == "1" && "$UPLOAD_CUML" == "1" ]]; then
  test -e ${CUML_FILE}
  echo "Upload cuml"
  echo ${CUML_FILE}
  gpuci_retry anaconda -t ${MY_UPLOAD_KEY} upload -u ${CONDA_USERNAME:-rapidsai} ${LABEL_OPTION} --skip-existing ${CUML_FILE}
fi


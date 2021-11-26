#!/bin/bash
# Copyright (c) 2018-2021, NVIDIA CORPORATION.
##############################################
# cuML CPU conda build script for CI         #
##############################################
set -ex

# Set path and build parallel level
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-8}

# Set home to the job's workspace
export HOME="$WORKSPACE"

# Determine CUDA release version
export CUDA_REL=${CUDA_VERSION%.*}

 # Setup 'gpuci_conda_retry' for build retries (results in 2 total attempts)
export GPUCI_CONDA_RETRY_MAX=1
export GPUCI_CONDA_RETRY_SLEEP=30

# Use Ninja to build, setup Conda Build Dir
export CMAKE_GENERATOR="Ninja"
export CONDA_BLD_DIR="$WORKSPACE/.conda-bld"

# Switch to project root; also root of repo checkout
cd "$WORKSPACE"

# If nightly build, append current YYMMDD to version
if [ "${IS_STABLE_BUILD}" != "true" ] ; then
  export VERSION_SUFFIX=`date +%y%m%d`
fi

################################################################################
# SETUP - Check environment
################################################################################

gpuci_logger "Check environment variables"
env

gpuci_logger "Activate conda env"
. /opt/conda/etc/profile.d/conda.sh
conda activate rapids

# Remove rapidsai-nightly channel if it is stable build
if [ "${IS_STABLE_BUILD}" = "true" ]; then
  conda config --system --remove channels rapidsai-nightly
fi

gpuci_logger "Check compiler versions"
python --version
$CC --version
$CXX --version

gpuci_logger "Check conda environment"
conda info
conda config --show-sources
conda list --show-channel-urls

# FIX Added to deal with Anancoda SSL verification issues during conda builds
conda config --set ssl_verify False

# FIXME: for now, force the building of all packages so they are built on a
# machine with a single CUDA version, then have the gpu/build.sh script simply
# install. This should eliminate a mismatch between different CUDA versions on
# cpu vs. gpu builds that is problematic with CUDA 11.5 Enhanced Compat.
BUILD_LIBCUML=1
BUILD_CUML=1

################################################################################
# BUILD - Conda package builds (conda deps: libcuml <- cuml)
################################################################################

if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
  if [ "$BUILD_LIBCUML" == '1' -o "$BUILD_CUML" == '1' ]; then
    gpuci_logger "Build conda pkg for libcuml"
    gpuci_conda_retry build --no-build-id --croot ${CONDA_BLD_DIR} conda/recipes/libcuml
  fi
else
  if [ "$BUILD_LIBCUML" == '1' ]; then
    gpuci_logger "PROJECT FLASH: Build conda pkg for libcuml"
    gpuci_conda_retry build --no-build-id --croot ${CONDA_BLD_DIR} conda/recipes/libcuml --dirty --no-remove-work-dir
    mkdir -p ${CONDA_BLD_DIR}/libcuml/work
    cp -r ${CONDA_BLD_DIR}/work/* ${CONDA_BLD_DIR}/libcuml/work
    rm -rf ${CONDA_BLD_DIR}/work
  fi
fi

if [ "$BUILD_CUML" == '1' ]; then
  if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    gpuci_logger "Build conda pkg for cuml"
    gpuci_conda_retry build --croot ${CONDA_BLD_DIR} conda/recipes/cuml --python=${PYTHON}
  else
    gpuci_logger "PROJECT FLASH: Build conda pkg for cuml"
    gpuci_conda_retry build --no-build-id --croot ${CONDA_BLD_DIR} conda/recipes/cuml -c $CONDA_BLD_DIR --dirty --no-remove-work-dir --python=${PYTHON}
    mkdir -p ${CONDA_BLD_DIR}/cuml/work
    cp -r ${CONDA_BLD_DIR}/work/* ${CONDA_BLD_DIR}/cuml/work
    rm -rf ${CONDA_BLD_DIR}/work
  fi
fi

################################################################################
# UPLOAD - Conda packages
################################################################################

gpuci_logger "Upload conda pkgs"
source ci/cpu/upload.sh


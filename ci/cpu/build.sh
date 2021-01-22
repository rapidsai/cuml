#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION.
##############################################
# cuML CPU conda build script for CI         # 
##############################################
set -ex

# Set path and build parallel level
export PATH=/opt/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=${PARALLEL_LEVEL:-4}

# Set home to the job's workspace
export HOME=$WORKSPACE

# Determine CUDA release version
export CUDA_REL=${CUDA_VERSION%.*}

 # Setup 'gpuci_conda_retry' for build retries (results in 2 total attempts)
export GPUCI_CONDA_RETRY_MAX=1
export GPUCI_CONDA_RETRY_SLEEP=30

# Switch to project root; also root of repo checkout
cd $WORKSPACE

# If nightly build, append current YYMMDD to version
if [[ "$BUILD_MODE" = "branch" && "$SOURCE_BRANCH" = branch-* ]] ; then
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

################################################################################
# BUILD - Conda package builds (conda deps: libcuml <- cuml)
################################################################################

if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
  if [ "$BUILD_LIBCUML" == '1' -o "$BUILD_CUML" == '1' ]; then
    gpuci_logger "Build conda pkg for libcuml"
    gpuci_conda_retry build conda/recipes/libcuml
  fi
else
  if [ "$BUILD_LIBCUML" == '1' ]; then
    gpuci_logger "PROJECT FLASH: Build conda pkg for libcuml"
    gpuci_conda_retry build conda/recipes/libcuml --dirty --no-remove-work-dir
  fi
fi

if [ "$BUILD_CUML" == '1' ]; then
  if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    gpuci_logger "Build conda pkg for cuml"
    gpuci_conda_retry build conda/recipes/cuml --python=${PYTHON}
  else
    gpuci_logger "PROJECT FLASH: Build conda pkg for cuml"
    gpuci_conda_retry build -c ci/artifacts/cuml/cpu/conda-bld/ --dirty --no-remove-work-dir conda/recipes/cuml --python=${PYTHON}
  fi
fi

################################################################################
# UPLOAD - Conda packages
################################################################################

gpuci_logger "Upload conda pkgs"
source ci/cpu/upload.sh


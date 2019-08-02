#!/bin/bash
# Copyright (c) 2018-2019, NVIDIA CORPORATION.
#########################################
# cuML GPU build and test script for CI #
#########################################
set -e
NUMARGS=$#
ARGS=$*

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Arg parsing function
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Set path and build parallel level
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4
export CUDA_REL=${CUDA_VERSION%.*}

# Set versions of packages needed to be grabbed
export CUDF_VERSION=0.8.*
export NVSTRINGS_VERSION=0.8.*
export RMM_VERSION=0.8.*

# Set home to the job's workspace
export HOME=$WORKSPACE

################################################################################
# SETUP - Check environment
################################################################################

logger "Check environment..."
env

logger "Check GPU usage..."
nvidia-smi

logger "Activate conda env..."
source activate gdf
conda install -c conda-forge -c rapidsai -c rapidsai-nightly -c nvidia \
      cudf=${CUDF_VERSION} \
      rmm=${RMM_VERSION} \
      nvstrings=${NVSTRINGS_VERSION} \
      lapack \
      cmake==3.14.3 \
      umap-learn \
      nccl>=2.4 \
      dask \
      distributed \
      dask-cudf \
      dask-cuda \
      statsmodels

# installing libclang separately so it doesn't get installed from conda-forge
conda install -c rapidsai \
      libclang

logger "Check versions..."
python --version
$CC --version
$CXX --version
conda list

################################################################################
# BUILD - Build libcuml, cuML, and prims from source
################################################################################

logger "Build libcuml..."
$WORKSPACE/build.sh clean libcuml cuml prims -v

################################################################################
# TEST - Run GoogleTest and py.tests for libcuml and cuML
################################################################################

if hasArg --skip-tests; then
    logger "Skipping Tests..."
    exit 0
fi

logger "Check GPU usage..."
nvidia-smi

logger "GoogleTest for libcuml..."
cd $WORKSPACE/cpp/build
GTEST_OUTPUT="xml:${WORKSPACE}/test-results/libcuml_cpp/" ./test/ml

logger "Python pytest for cuml..."
cd $WORKSPACE/python
pytest --cache-clear --junitxml=${WORKSPACE}/junit-cuml.xml -v -m "not mg"

################################################################################
# TEST - Run GoogleTest for ml-prims
################################################################################

logger "Run ml-prims test..."
cd $WORKSPACE/cpp/build
GTEST_OUTPUT="xml:${WORKSPACE}/test-results/prims/" ./test/prims

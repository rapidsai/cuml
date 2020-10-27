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

# Parse git describe
cd $WORKSPACE
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`

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
conda install -c conda-forge -c rapidsai -c rapidsai-nightly/label/testing -c rapidsai-nightly -c nvidia \
      "cupy>7.1.0,<9.0.0a0" \
      "cudatoolkit=${CUDA_REL}" \
      "cudf=${MINOR_VERSION}" \
      "rmm=${MINOR_VERSION}" \
      "libcumlprims=${MINOR_VERSION}" \
      "lapack" \
      "cmake==3.14.3" \
      "umap-learn" \
      "nccl>=2.5" \
      "dask>=2.12.0" \
      "distributed>=2.12.0" \
      "dask-cudf=${MINOR_VERSION}" \
      "dask-cuda=${MINOR_VERSION}" \
      "ucx-py=${MINOR_VERSION}" \
      "statsmodels" \
      "xgboost==1.1.0dev.rapidsai0.15" \
      "lightgbm"

logger "Check versions..."
python --version
$CC --version
$CXX --version
conda list

################################################################################
# BUILD - Build libcuml++, cuML, and prims from source
################################################################################

logger "Build libcuml++..."
$WORKSPACE/build.sh clean libcuml cuml prims bench -v

################################################################################
# TEST - Run MG GoogleTest and py.tests for libcuml++ and cuML
################################################################################

if hasArg --skip-tests; then
    logger "Skipping Tests..."
    exit 0
fi

logger "Check GPU usage..."
nvidia-smi

# Disabled while CI/the test become compatible
# logger "MG GoogleTest for libcuml mg..."
# cd $WORKSPACE/cpp/build
# GTEST_OUTPUT="xml:${WORKSPACE}/test-results/libcuml_cpp_mg/" ./test/ml_mg

logger "Python MG pytest for cuml..."
cd $WORKSPACE/python
pytest --cache-clear --junitxml=${WORKSPACE}/junit-cuml.xml -v -m "mg"

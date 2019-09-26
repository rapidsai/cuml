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

# Set home to the job's workspace
export HOME=$WORKSPACE

# Parse git describei
cd $WORKSPACE
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`

################################################################################
# SETUP - Check environment
################################################################################

logger "Check environment..."
env

logger "Check GPU usage..."
nvidia-smi

logger "Activate conda env..."
source activate gdf
conda install -c conda-forge -c rapidsai -c rapidsai-nightly -c rapidsai/label/xgboost -c nvidia \
      "rapidsai/label/cuda${CUDA_REL}::cupy>=6.2" \
      "cudatoolkit=${CUDA_REL}" \
      "cudf=${MINOR_VERSION}" \
      "rmm=${MINOR_VERSION}" \
      "nvstrings=${MINOR_VERSION}" \
      "libcumlprims=${MINOR_VERSION}" \
      "lapack" \
      "cmake==3.14.3" \
      "umap-learn" \
      "nccl>=2.4" \
      "dask=2.3.0" \
      "distributed=2.3.0" \
      "dask-ml" \
      "dask-cudf=${MINOR_VERSION}" \
      "dask-cuda=${MINOR_VERSION}" \
      "statsmodels" \
      "xgboost=0.90.rapidsdev1"


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

logger "Adding ${CONDA_PREFIX}/lib to LD_LIBRARY_PATH"

export LD_LIBRARY_PATH_CACHED=$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

logger "Build libcuml..."
$WORKSPACE/build.sh clean libcuml cuml prims --multigpu -v

logger "Resetting LD_LIBRARY_PATH..."

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH_CACHED
export LD_LIBRARY_PATH_CACHED=""

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
pytest --cache-clear --junitxml=${WORKSPACE}/junit-cuml.xml -v --ignore=cuml/test/test_trustworthiness.py

################################################################################
# TEST - Run GoogleTest for ml-prims
################################################################################

logger "Run ml-prims test..."
cd $WORKSPACE/cpp/build
GTEST_OUTPUT="xml:${WORKSPACE}/test-results/prims/" ./test/prims

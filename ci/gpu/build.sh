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
conda install -c conda-forge -c rapidsai -c rapidsai-nightly -c nvidia \
      "cupy>=7,<8.0.0a0" \
      "cudatoolkit=${CUDA_REL}" \
      "cudf=${MINOR_VERSION}" \
      "rmm=${MINOR_VERSION}" \
      "nvstrings=${MINOR_VERSION}" \
      "libcumlprims=${MINOR_VERSION}" \
      "lapack" \
      "cmake==3.14.3" \
      "umap-learn" \
      "protobuf>=3.4.1,<4.0.0" \
      "nccl>=2.5" \
      "dask>=2.12.0" \
      "distributed>=2.12.0" \
      "dask-cudf=${MINOR_VERSION}" \
      "dask-cuda=${MINOR_VERSION}" \
      "ucx-py=${MINOR_VERSION}" \
      "statsmodels" \
      "xgboost====1.0.2dev.rapidsai0.13" \
      "lightgbm"


# Install the master version of dask, distributed, and dask-ml
logger "pip install git+https://github.com/dask/distributed.git --upgrade --no-deps"
pip install "git+https://github.com/dask/distributed.git" --upgrade --no-deps
logger "pip install git+https://github.com/dask/dask.git --upgrade --no-deps"
pip install "git+https://github.com/dask/dask.git" --upgrade --no-deps


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

logger "Build libcuml, cuml, prims and bench targets..."
$WORKSPACE/build.sh clean libcuml cuml prims bench -v

logger "Resetting LD_LIBRARY_PATH..."

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH_CACHED
export LD_LIBRARY_PATH_CACHED=""

logger "Build treelite for GPU testing..."
# Buildint treelite Python for testing is temporary while there is a pip/conda
# treelite package

cd $WORKSPACE/cpp/build/treelite/src/treelite
mkdir build
cd build
cmake ..
make -j${PARALLEL_LEVEL}
cd ../python
python setup.py install

cd $WORKSPACE


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

pytest --cache-clear --junitxml=${WORKSPACE}/junit-cuml.xml -v -s -m "not memleak"

################################################################################
# TEST - Run GoogleTest for ml-prims
################################################################################

logger "Run ml-prims test..."
cd $WORKSPACE/cpp/build
GTEST_OUTPUT="xml:${WORKSPACE}/test-results/prims/" ./test/prims

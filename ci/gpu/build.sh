#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION.
#########################################
# cuML GPU build and test script for CI #
#########################################
set -e

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Set path and build parallel level
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4

# Set home to the job's workspace
export HOME=$WORKSPACE

################################################################################
# SETUP - Check environment
################################################################################

logger "Check environment..."
env

logger "Check GPU usage..."
nvidia-smi

logger "Check versions..."
python --version
$CC --version
$CXX --version
conda list

################################################################################
# BUILD - Build libcuml and cuML from source
################################################################################

logger "Build libcuml..."
mkdir -p $WORKSPACE/cuML/build
cd $WORKSPACE/cuML/build
logger "Run cmake libcuml..."
cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_CXX11_ABI=ON ..

logger "Clean up make..."
make clean

logger "Make libcuml..."
make -j${PARALLEL_LEVEL}

logger "Install libcuml..."
make -j${PARALLEL_LEVEL} install


logger "Build cuML..."
cd $WORKSPACE/python
python setup.py build_ext --inplace

################################################################################
# TEST - Run GoogleTest and py.tests for libcuml and cuML
################################################################################

logger "Check GPU usage..."
nvidia-smi

logger "GoogleTest for libcuml..."
cd $WORKSPACE/cuML/build
GTEST_OUTPUT="xml:${WORKSPACE}/test-results/" ./ml_test


logger "Python py.test for cuML..."
cd $WORKSPACE/python
py.test --cache-clear --junitxml=${WORKSPACE}/junit-cuml.xml -v

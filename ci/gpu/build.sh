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
export CUDA_REL=${CUDA_VERSION%.*}

# Set versions of packages needed to be grabbed
export CUDF_VERSION=0.7.*
export NVSTRINGS_VERSION=0.7.*
export RMM_VERSION=0.7.*

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
conda install -c rapidsai/label/cuda${CUDA_REL} -c rapidsai-nightly/label/cuda${CUDA_REL} cudf=${CUDF_VERSION} rmm=${RMM_VERSION} nvstrings=${NVSTRINGS_VERSION}
conda install -c conda-forge lapack cmake==3.14.3 umap-learn

logger "Check versions..."
python --version
$CC --version
$CXX --version
conda list

logger "Check GPU running the tests..."
GPU="$(nvidia-smi | awk '{print $4}' | sed '8!d')"
echo "Running tests on $GPU"

if [[ $GPU == *"P100"* ]]; then
  logger "Building for Pascal..."
  GPU_ARCH="-DGPU_ARCHS=\"60\""
elif [[ $GPU == *"V100"* ]]; then
  logger "Building for Volta..."
  GPU_ARCH=GPU_ARCH="-DGPU_ARCHS=\"70\""
elif [[ $GPU == *"T4"* ]]; then
  logger "Building for Turing..."
  GPU_ARCH=GPU_ARCH="-DGPU_ARCHS=\"75\""
fi

################################################################################
# BUILD - Build libcuml and cuML from source
################################################################################

logger "Build libcuml..."
mkdir -p $WORKSPACE/cuML/build
cd $WORKSPACE/cuML/build
logger "Run cmake libcuml..."
cmake -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX -DCMAKE_CXX11_ABI=ON -DBLAS_LIBRARIES=$CONDA_PREFIX/lib/libopenblas.a -DLAPACK_LIBRARIES=$CONDA_PREFIX/lib/libopenblas.a $GPU_ARCH ..

logger "Clean up make..."
make clean

logger "Make libcuml..."
make -j${PARALLEL_LEVEL}

logger "Install libcuml..."
make -j${PARALLEL_LEVEL} install

logger "Build cuML..."
cd $WORKSPACE/python
python setup.py build_ext --inplace

logger "Build ml-prims tests..."
mkdir -p $WORKSPACE/ml-prims/build
cd $WORKSPACE/ml-prims/build
cmake $GPU_ARCH ..

logger "Clean up make..."
make clean
logger "Make ml-prims test..."
make -j${PARALLEL_LEVEL}


################################################################################
# TEST - Run GoogleTest and py.tests for libcuml and cuML
################################################################################

logger "Check GPU usage..."
nvidia-smi

logger "GoogleTest for libcuml..."
cd $WORKSPACE/cuML/build
GTEST_OUTPUT="xml:${WORKSPACE}/test-results/libcuml_cpp/" ./ml_test


logger "Python py.test for cuML..."
cd $WORKSPACE/python
py.test --cache-clear --junitxml=${WORKSPACE}/junit-cuml.xml -v


logger "Run ml-prims test..."
cd $WORKSPACE/ml-prims/build
GTEST_OUTPUT="xml:${WORKSPACE}/test-results/ml-prims/" ./test/mlcommon_test

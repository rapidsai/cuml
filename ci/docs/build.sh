#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION.
#########################################
# cuML GPU build and test script for CI #
#########################################
set -ex

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Set path and build parallel level
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH
export PARALLEL_LEVEL=4
export CUDA_REL=${CUDA_VERSION%.*}
export CUDF_VERSION=0.7.*
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
conda install -c rapidsai/label/cuda$CUDA_REL -c rapidsai-nightly/label/cuda$CUDA_REL -c conda-forge \
    cudf=$CUDF_VERSION rmm=$RMM_VERSION

pip install numpydoc sphinx sphinx-rtd-theme sphinxcontrib-websupport

logger "Check versions..."
python --version
$CC --version
$CXX --version
conda list

################################################################################
# BUILD - Build libcuml and cuML from source
################################################################################

cd $WORKSPACE
git submodule update --init --recursive

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
python setup.py install

################################################################################
# BUILD - Build docs
################################################################################

logger "Build docs..."
cd $WORKSPACE/docs
make html

rm -rf /data/docs/html/*
mv build/html/* /data/docs/html

#!/bin/bash
# Copyright (c) 2018-2020, NVIDIA CORPORATION.
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
export CUDF_VERSION=0.8.*
export RMM_VERSION=0.8.*

# Set home to the job's workspace
export HOME=$WORKSPACE
export DOCS_DIR=/data/docs/html

while getopts "d" option; do
    case ${option} in
        d)
            DOCS_DIR=${OPTARG}
            ;;
    esac
done

################################################################################
# SETUP - Check environment
################################################################################

logger "Check environment..."
env

logger "Check GPU usage..."
nvidia-smi

logger "Activate conda env..."
source activate gdf
conda install -c nvidia -c rapidsai -c rapidsai-nightly -c conda-forge \
    cudf=$CUDF_VERSION rmm=$RMM_VERSION cudatoolkit=$CUDA_REL

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
$WORKSPACE/build.sh clean libcuml cuml

################################################################################
# BUILD - Build doxygen docs
################################################################################

cd $WORKSPACE/cpp/build
logger "Build doxygen docs..."
make doc

################################################################################
# BUILD - Build docs
################################################################################

logger "Build docs..."
cd $WORKSPACE/docs
make html

rm -rf ${DOCS_DIR}/*
mv build/html/* $DOCS_DIR

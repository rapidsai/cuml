#!/bin/bash

# Copyright (c) 2018-2021, NVIDIA CORPORATION.
##############################################
# cuML local build and test script for CI    #
##############################################


##############################################
# User defined variables                     #
##############################################

CUDA_TEST_VERSION=${CUDA_TEST_VERSION:=11.0}
PARALLEL_LEVEL=${PARALLEL_LEVEL:=8}
PYTHON_VERSION=DEFAULT
CUML_PATH=${CUML_PATH:=$PWD/../..}
REPO_PATH=${REPO_PATH:="$(dirname "$(dirname "$PWD")")"}
CONTAINER_PYTHON_VERSION}


##############################################
# Derived variables
##############################################

GIT_DESCRIBE_TAG=`git describe --tags`
MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`
BUILD_DOCKER="gpuci/rapidsai-driver:${MINOR_VERSION}-cuda11.5-devel-centos7-py3.7"
TEST_CONTAINER="gpuci/rapidsai:${MINOR_VERSION}-cuda${CUDA_TEST_VERSION}-devel-centos7-py3.7"

BASE_CONTAINER_BUILD_DIR=${REPO_PATH}/build_$(echo $(basename "${BUILD_DOCKER}")|sed -e 's/:/_/g')



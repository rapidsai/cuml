#!/bin/bash
# Copyright (c) 2018-2023, NVIDIA CORPORATION.

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS} -ccbin ${CXX}" # Needed for CUDA 12 nvidia channel compilers
./build.sh -n libcuml prims -v --allgpuarch

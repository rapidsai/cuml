#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Usage: bash apply_wheel_modifications.sh <new_version>

VERSION=${1}
CUDA_SUFFIX=${2}

# pyproject.toml versions
sed -i "s/^version = .*/version = \"${VERSION}\"/g" python/pyproject.toml

# pyproject.toml cuda suffixes
sed -i "s/^name = \"cuml\"/name = \"cuml${CUDA_SUFFIX}\"/g" python/pyproject.toml
sed -i "s/cudf/cudf${CUDA_SUFFIX}/g" python/pyproject.toml
sed -i "s/pylibraft/pylibraft${CUDA_SUFFIX}/g" python/pyproject.toml
sed -i "s/raft-dask/raft-dask${CUDA_SUFFIX}/g" python/pyproject.toml
sed -i "s/rmm/rmm${CUDA_SUFFIX}/g" python/pyproject.toml

if [[ $CUDA_SUFFIX == "-cu12" ]]; then
    sed -i "s/cuda-python[<=>\.,0-9]*/cuda-python>=12.0,<13.0/g" python/pyproject.toml
    sed -i "s/cupy-cuda11x/cupy-cuda12x/g" python/pyproject.toml
    sed -i "s/numba[<=>\.,0-9]*/numba>=0.57/g" python/pyproject.toml
fi

#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Usage: bash apply_wheel_modifications.sh <new_version>

VERSION=${1}
CUDA_SUFFIX=${2}

# __init__.py versions
sed -i "s/__version__ = .*/__version__ = \"${VERSION}\"/g" python/cuml/__init__.py

# pyproject.toml versions
sed -i "s/^version = .*/version = \"${VERSION}\"/g" python/pyproject.toml

# pyproject.toml cuda suffixes
sed -i "s/^name = \"cuml\"/name = \"cuml${CUDA_SUFFIX}\"/g" python/pyproject.toml
sed -i "s/cudf/cudf${CUDA_SUFFIX}/g" python/pyproject.toml
sed -i "s/pylibraft/pylibraft${CUDA_SUFFIX}/g" python/pyproject.toml
sed -i "s/raft-dask/raft-dask${CUDA_SUFFIX}/g" python/pyproject.toml
sed -i "s/rmm/rmm${CUDA_SUFFIX}/g" python/pyproject.toml

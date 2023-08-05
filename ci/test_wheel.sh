#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="cuml_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 ./dist

# On arm also need to install CMake because treelite needs to be compiled (no wheels available for arm).
if [[ "$(arch)" == "aarch64" ]]; then
    python -m pip install cmake
fi

# Always install latest dask for testing
python -m pip install git+https://github.com/dask/dask.git@2023.7.1 git+https://github.com/dask/distributed.git@2023.7.1 git+https://github.com/rapidsai/dask-cuda.git@branch-23.10

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install $(echo ./dist/cuml*.whl)[test]

# Run smoke tests for aarch64 pull requests
if [[ "$(arch)" == "aarch64" && "${RAPIDS_BUILD_TYPE}" == "pull-request" ]]; then
    python ci/wheel_smoke_test_cuml.py
else
    python -m pytest ./python/cuml/tests -k 'not test_sparse_pca_inputs' -n 8 --ignore=python/cuml/tests/dask && python -m pytest ./python/cuml/tests -k 'test_sparse_pca_inputs' && python -m pytest ./python/cuml/tests/dask
fi

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

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install $(echo ./dist/cuml*.whl)[test]

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Run smoke tests for aarch64 pull requests
if [[ "$(arch)" == "aarch64" && "${RAPIDS_BUILD_TYPE}" == "pull-request" ]]; then
    python ci/wheel_smoke_test_cuml.py
else
    rapids-logger "pytest cuml single GPU"
    ./ci/run_cuml_singlegpu_pytests.sh \
      --numprocesses=8 \
      --dist=worksteal \
      -k 'not test_sparse_pca_inputs' \
      --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml.xml"

    # Run test_sparse_pca_inputs separately
    ./ci/run_cuml_singlegpu_pytests.sh \
      -k 'test_sparse_pca_inputs' \
      --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml-sparse-pca.xml"

    rapids-logger "pytest cuml-dask"
    ./ci/run_cuml_dask_pytests.sh \
      --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml-dask.xml"
fi

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="cuml_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist
RAPIDS_PY_WHEEL_NAME="libcuml_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp ./dist

# echo to expand wildcard before adding `[extra]` requires for pip
rapids-pip-retry install \
  ./dist/libcuml*.whl \
  "$(echo ./dist/cuml*.whl)[test]"

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
mkdir -p "${RAPIDS_TESTS_DIR}"

EXITCODE=0
trap "EXITCODE=1" ERR
set +e


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

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

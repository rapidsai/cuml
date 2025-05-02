#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
CUML_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cuml_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)
LIBCUML_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcuml_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
mkdir -p "${RAPIDS_TESTS_DIR}"

# Install just minimal dependencies first
rapids-pip-retry install \
  "${LIBCUML_WHEELHOUSE}"/libcuml*.whl \
  "${CUML_WHEELHOUSE}"/cuml*.whl

# Try to import cuml with just a minimal install"
rapids-logger "Importing cuml with minimal dependencies"
python -c "import cuml"

# echo to expand wildcard before adding `[extra]` requires for pip
rapids-pip-retry install \
   "${LIBCUML_WHEELHOUSE}"/libcuml*.whl \
  "$(echo "${CUML_WHEELHOUSE}"/cuml*.whl)[test]"

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

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

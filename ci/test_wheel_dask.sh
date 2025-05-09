#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

set -euo pipefail

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
CUML_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cuml_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)
LIBCUML_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcuml_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
mkdir -p "${RAPIDS_TESTS_DIR}"

# echo to expand wildcard before adding `[extra]` requires for pip
rapids-pip-retry install \
   "${LIBCUML_WHEELHOUSE}"/libcuml*.whl \
  "$(echo "${CUML_WHEELHOUSE}"/cuml*.whl)[dask,test,test-dask]"

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest cuml dask"
timeout 2h ./ci/run_cuml_dask_pytests.sh --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml-dask.xml"

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

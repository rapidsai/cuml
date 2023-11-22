#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Generate Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file_key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)
LIBRMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1374 cpp)
RMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1374 python)
LIBRAFT_CHANNEL=$(rapids-get-pr-conda-artifact raft 1957 cpp)
RAFT_CHANNEL=$(rapids-get-pr-conda-artifact raft 1957 python)
LIBCUDF_CHANNEL=$(rapids-get-pr-conda-artifact cudf 14355 cpp)
CUDF_CHANNEL=$(rapids-get-pr-conda-artifact cudf 14355 python)

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

rapids-print-env

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${PYTHON_CHANNEL}" \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${RMM_CHANNEL}" \
  --channel "${LIBRAFT_CHANNEL}" \
  --channel "${RAFT_CHANNEL}" \
  --channel "${LIBCUDF_CHANNEL}" \
  --channel "${CUDF_CHANNEL}" \
  libcuml cuml

rapids-logger "Check GPU usage"
nvidia-smi

# Enable hypothesis testing for nightly test runs.
if [ "${RAPIDS_BUILD_TYPE}" == "nightly" ]; then
  export HYPOTHESIS_ENABLED="true"
fi

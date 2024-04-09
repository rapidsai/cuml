#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

# Support invoking test_python_dask.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

# Common setup steps shared by Python test jobs
source ./ci/test_python_common.sh

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest cuml-dask"

# Run tests (no UCX-Py/UCXX)
./ci/run_cuml_dask_pytests.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml-dask.xml" \
  --cov-config=../../../.coveragerc \
  --cov=cuml_dask \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cuml-dask-coverage.xml" \
  --cov-report=term

# Run tests (UCX-Py only)
./ci/run_cuml_dask_pytests.sh \
  --run_ucx \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml-dask-ucx.xml" \
  --cov-config=../../../.coveragerc \
  --cov=cuml_dask \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cuml-dask-ucx-coverage.xml" \
  --cov-report=term \
  .

# Run tests (UCXX only)
./ci/run_cuml_dask_pytests.sh \
  --run_ucxx \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml-dask-ucxx.xml" \
  --cov-config=../../../.coveragerc \
  --cov=cuml_dask \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cuml-dask-ucxx-coverage.xml" \
  --cov-report=term \
  .

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

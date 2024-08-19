#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

# Support invoking test_python_dask.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

# Common setup steps shared by Python test jobs
source ./ci/test_python_common.sh

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

test_args=(
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml-dask.xml"
  --cov-config=../../../.coveragerc
  --cov=cuml_dask
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cuml-dask-coverage.xml"
  --cov-report=term
)

# Run tests
rapids-logger "pytest cuml-dask (No UCX-Py/UCXX)"
timeout 2h ./ci/run_cuml_dask_pytests.sh "${test_args[@]}"

rapids-logger "pytest cuml-dask (UCX-Py only)"
timeout 5m ./ci/run_cuml_dask_pytests.sh "${test_args[@]}" --run_ucx

rapids-logger "pytest cuml-dask (UCXX only)"
timeout 5m ./ci/run_cuml_dask_pytests.sh "${test_args[@]}" --run_ucxx

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

#!/bin/bash
# Copyright (c) 2022-2025, NVIDIA CORPORATION.

# Support invoking test_python_singlegpu.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../ || exit 1

# Common setup steps shared by Python test jobs
source ./ci/test_python_common.sh

# We want to error if dask is installed in this environment so
if python -c 'import dask' 2>/dev/null; then
  echo "ERROR: dask is installed in this environment! This means we are not \
  testing that cuml works without dask. Please adjust the environment so the \
  main test environment doesn't install dask."
  exit 1
fi

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest cuml single GPU"
./ci/run_cuml_singlegpu_pytests.sh \
  --numprocesses=8 \
  --dist=worksteal \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml.xml" \
  --cov-config=../.coveragerc \
  --cov=cuml \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cuml-coverage.xml"

  rapids-logger "pytest cuml accelerator"
./ci/run_cuml_singlegpu_accel_pytests.sh \
  --numprocesses=8 \
  --dist=worksteal \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml-accel.xml" \
  --cov-config=../.coveragerc \
  --cov=cuml \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cuml-accel-coverage.xml"

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

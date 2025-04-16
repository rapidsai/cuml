#!/bin/bash
# Copyright (c) 2022-2025, NVIDIA CORPORATION.

# Support invoking test_python_singlegpu.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../ || exit 1

# Common setup steps shared by Python test jobs
source ./ci/test_python_common.sh

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest cuml single GPU"
./ci/run_cuml_singlegpu_pytests.sh \
  --numprocesses=8 \
  --dist=worksteal \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml.xml" \
  --cov-config=../../.coveragerc \
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


if [ "${RAPIDS_BUILD_TYPE}" == "nightly" ]; then
  rapids-logger "memory leak pytests"

  ./ci/run_cuml_singlegpu_memleak_pytests.sh \
    --numprocesses=1 \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml-memleak.xml" \
    --cov-config=../../.coveragerc \
    --cov=cuml \
    --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cuml-memleak-coverage.xml" \
    -m "memleak"
fi

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

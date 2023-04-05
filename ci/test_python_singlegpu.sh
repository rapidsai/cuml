#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

# Common setup steps shared by Python test jobs
source "$(dirname "$0")/test_python_common.sh"

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest cuml single GPU..."
cd python/cuml/tests
pytest \
  --numprocesses=8 \
  --ignore=dask \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml.xml" \
  --cov-config=../../.coveragerc \
  --cov=cuml \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cuml-coverage.xml" \
  --cov-report=term \
  -m "not memleak" \
  .

  rapids-logger "memory leak pytests..."

pytest \
  --numprocesses=1 \
  --ignore=dask \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml-memleak.xml" \
  --cov-config=../../.coveragerc \
  --cov=cuml \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cuml-memleak-coverage.xml" \
  --cov-report=term \
  -m "memleak" \
  .

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

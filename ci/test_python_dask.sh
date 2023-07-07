#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

# Common setup steps shared by Python test jobs
source "$(dirname "$0")/test_python_common.sh"

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest cuml-dask"
cd python/cuml/tests/dask
pytest \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml-dask.xml" \
  --cov-config=../../../.coveragerc \
  --cov=cuml_dask \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cuml-dask-coverage.xml" \
  --cov-report=term \
  .

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

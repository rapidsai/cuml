#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

# Common setup steps shared by Python test jobs
source "$(dirname "$0")/test_python_common.sh"

rapids-logger "pytest cuml-dask"
pushd python/cuml/tests/dask
pytest \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml-dask.xml" \
  --cov-config=../../../.coveragerc \
  --cov=cuml_dask \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cuml-dask-coverage.xml" \
  --cov-report=term \
  .
exitcode=$?

if (( ${exitcode} != 0 )); then
    SUITEERROR=${exitcode}
    echo "FAILED: 1 or more tests in cuml-dask"
fi
popd

exit ${SUITEERROR}

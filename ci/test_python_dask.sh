#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

# Common setup steps shared by Python test jobs
source "$(dirname "$0")/test_python_common.sh"

rapids-logger "pytest cuml-dask"
pytest \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml-dask.xml" \
  --cov-config=python/.coveragerc \
  --cov=cuml_dask \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cuml-dask-coverage.xml" \
  --cov-report=term \
  python/cuml/tests/dask

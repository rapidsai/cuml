#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

# Common setup steps shared by Python test jobs
source "$(dirname "$0")/test_python_common.sh"

rapids-logger "pytest cuml single GPU"
pytest \
  --numprocesses=8 \
  --ignore=python/cuml/tests/dask \
  --cache-clear \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml.xml" \
  --cov-config=../../.coveragerc \
  --cov=cuml \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cuml-coverage.xml" \
  --cov-report=term \
  python/cuml/tests

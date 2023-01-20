#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

# Common setup steps shared by Python test jobs
source "$(dirname "$0")/test_python_common.sh"

rapids-mamba-retry install -c conda-forge scikit-learn=1.2
pip install treelite==3.1

rapids-logger "pytest cuml single GPU"
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
  .

#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

# Support invoking test script outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../ || exit 1

# Common setup steps shared by Python test jobs
source ./ci/test_python_common.sh

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Run scikit-learn tests with acceleration enabled
rapids-logger "Running scikit-learn tests with cuML acceleration"

# Run the tests
./python/cuml/cuml/accel/tests/scikit-learn/run-tests.sh \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml-accel-scikit-learn.xml"

# Analyze results and check pass rate threshold
rapids-logger "Analyzing test results"
./python/cuml/cuml/accel/tests/scikit-learn/summarize-results.py \
    --fail-below 80 \
    "${RAPIDS_TESTS_DIR}/junit-cuml-accel-scikit-learn.xml"

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

# Support invoking test script outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

# Common setup steps shared by Python test jobs
source ./ci/test_python_common.sh

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Run scikit-learn tests with acceleration enabled
rapids-logger "Running scikit-learn tests with cuML acceleration"

# Run the tests
./ci/accel/scikit-learn-tests/run-tests.sh \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml-accel-scikit-learn.xml" || true

# Analyze results and check pass rate threshold
rapids-logger "Analyzing test results"
./ci/accel/scikit-learn-tests/summarize-results.sh \
    --fail-below 80 \
    "${RAPIDS_TESTS_DIR}/junit-cuml-accel-scikit-learn.xml"

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

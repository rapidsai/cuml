#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

# Support invoking test script outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../ || exit 1

# Common setup steps shared by Python test jobs
source ./ci/test_python_common.sh

# Run scikit-learn tests with acceleration enabled
rapids-logger "Running scikit-learn tests with cuML acceleration"

# Run the tests and capture the exit code
timeout 1h ./python/cuml/cuml/accel/tests/scikit-learn/run-tests.sh \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml-accel-scikit-learn.xml"
TEST_EXITCODE=$?

# Analyze results and check pass rate threshold
rapids-logger "Analyzing test results"
./python/cuml/cuml/accel/tests/scikit-learn/summarize-results.py \
    --fail-below 85 \
    "${RAPIDS_TESTS_DIR}/junit-cuml-accel-scikit-learn.xml"
THRESHOLD_EXITCODE=$?

# Set final exit code based on conditions
if [ "${RAPIDS_BUILD_TYPE}" == "nightly" ]; then
    # For nightly runs, only fail if threshold is not met
    EXITCODE=$THRESHOLD_EXITCODE
else
    # For regular runs, fail on any test failure
    EXITCODE=$TEST_EXITCODE
fi

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

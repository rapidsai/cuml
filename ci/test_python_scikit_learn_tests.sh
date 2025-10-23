#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

# Support invoking test script outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../ || exit 1

# Common setup steps shared by Python test jobs
source ./ci/test_python_common.sh

# Run scikit-learn tests with acceleration enabled
rapids-logger "Running scikit-learn tests with cuML acceleration"

# Do not immediately exit on error
set +e

# Run the tests and capture the exit code
timeout 1h ./python/cuml/cuml_accel_tests/upstream/scikit-learn/run-tests.sh \
    --numprocesses=8 \
    --dist=worksteal \
    --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml-accel-scikit-learn.xml"
TEST_EXITCODE=$?

# Analyze results and check pass rate threshold
rapids-logger "Analyzing test results"
./python/cuml/cuml_accel_tests/upstream/summarize-results.py \
    --config ./python/cuml/cuml_accel_tests/upstream/scikit-learn/test_config.yaml \
    "${RAPIDS_TESTS_DIR}/junit-cuml-accel-scikit-learn.xml"
THRESHOLD_EXITCODE=$?

# Set final exit code based on build type
if [ "${RAPIDS_BUILD_TYPE}" == "pull-request" ]; then
    # For pull requests, fail on any test failure
    EXITCODE=$TEST_EXITCODE
else
    # For all other builds, only fail if threshold is not met
    EXITCODE=$THRESHOLD_EXITCODE
fi

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

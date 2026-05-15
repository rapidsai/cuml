#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Support invoking test script outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../ || exit 1

# Common setup steps shared by Python test jobs
source ./ci/test_python_common.sh

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

SKLEARN_EXAMPLES_JUNITXML="${RAPIDS_TESTS_DIR}/junit-sklearn-examples.xml"
SKLEARN_EXAMPLES_MAX_XFAILED="${SKLEARN_EXAMPLES_MAX_XFAILED:-0}"
SKLEARN_EXAMPLES_MAX_XPASSED="${SKLEARN_EXAMPLES_MAX_XPASSED:-0}"

# Run scikit-learn examples under cuml.accel
rapids-logger "scikit-learn examples"
timeout -v --signal=SIGINT --kill-after=60s 60m ./python/cuml/cuml_accel_tests/upstream/scikit-learn/run-examples.sh \
    -n 4 --dist worksteal \
    --junitxml="${SKLEARN_EXAMPLES_JUNITXML}"
TEST_EXITCODE=$?

# The examples tests are still being ratcheted down to zero expected xfails,
# while also requiring a healthy majority of examples to pass.
rapids-logger "scikit-learn examples: enforce pass-rate and xfail ratchets"
./python/cuml/cuml_accel_tests/upstream/summarize-results.py \
    --fail-below 50 \
    --max-xfailed "${SKLEARN_EXAMPLES_MAX_XFAILED}" \
    --max-xpassed "${SKLEARN_EXAMPLES_MAX_XPASSED}" \
    "${SKLEARN_EXAMPLES_JUNITXML}"
SUMMARY_EXITCODE=$?

if [ "${TEST_EXITCODE}" != "0" ]; then
    EXITCODE="${TEST_EXITCODE}"
elif [ "${SUMMARY_EXITCODE}" != "0" ]; then
    EXITCODE="${SUMMARY_EXITCODE}"
fi

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

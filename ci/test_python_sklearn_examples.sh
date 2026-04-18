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

# Run scikit-learn examples under cuml.accel
rapids-logger "scikit-learn examples"
timeout 60m ./python/cuml/cuml_accel_tests/upstream/scikit-learn/run-examples.sh \
    -n 4 --dist worksteal \
    --junitxml="${SKLEARN_EXAMPLES_JUNITXML}"

# The examples tests tolerate timeouts and network issues (warn-only), but
# require a healthy majority of examples to pass so widespread regressions are
# not missed.
rapids-logger "scikit-learn examples: require >=50% pass rate"
./python/cuml/cuml_accel_tests/upstream/summarize-results.py \
    --fail-below 50 \
    "${SKLEARN_EXAMPLES_JUNITXML}"

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Support invoking test script outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../ || exit 1

# Common setup steps shared by Python test jobs
export DEPENDENCY_FILE_KEY=test_python_accel_sklearn
source ./ci/test_python_common.sh

rapids-logger "Install optional scikit-learn example dependencies"
rapids-mamba-retry install --yes -n test \
    "plotly>=6,<7" \
    "polars>=1,<2" \
    "pooch>=1,<2" \
    scikit-image

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

SKLEARN_EXAMPLES_JUNITXML="${RAPIDS_TESTS_DIR}/junit-sklearn-examples.xml"

# Run scikit-learn examples under cuml.accel
rapids-logger "scikit-learn examples"
timeout -v --signal=SIGINT --kill-after=60s 60m ./python/cuml/cuml_accel_tests/upstream/scikit-learn/run-examples.sh \
    -vv --durations=0 --durations-min=0 \
    -n 4 --dist worksteal \
    --example-timeout=300 \
    --junitxml="${SKLEARN_EXAMPLES_JUNITXML}"

# Per-example timeouts and network failures are reported as xfails. The
# examples tests still require a healthy majority of examples to pass so
# widespread regressions are not missed.
rapids-logger "scikit-learn examples: require >=90% pass rate"
./python/cuml/cuml_accel_tests/upstream/summarize-results.py \
    --fail-below 90 \
    "${SKLEARN_EXAMPLES_JUNITXML}"

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

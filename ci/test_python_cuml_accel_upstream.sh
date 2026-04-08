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

# Run scikit-learn examples under cuml.accel
rapids-logger "scikit-learn examples"
timeout 60m ./python/cuml/cuml_accel_tests/upstream/scikit-learn/run-examples.sh \
  -n auto --dist worksteal \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-sklearn-examples.xml"

# Run UMAP tests
rapids-logger "UMAP test suite"
timeout 15m ./python/cuml/cuml_accel_tests/upstream/umap/run-tests.sh

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

# Support invoking test script outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../ || exit 1

# Common setup steps shared by Python test jobs
source ./ci/test_python_common.sh

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Run UMAP tests
rapids-logger "UMAP test suite"
timeout 15m ./python/cuml/cuml_accel_tests/upstream/umap/run-tests.sh

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Support invoking test_python_dask.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../ || exit

# Common setup steps shared by Python test jobs
export DEPENDENCY_FILE_KEY=test_python_dask
source ./ci/test_python_common.sh

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

test_args=(
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml-dask.xml"
)

# Run tests
rapids-logger "pytest cuml-dask (No UCXX)"
timeout -v 1h ./ci/run_cuml_dask_pytests.sh "${test_args[@]}"

rapids-logger "pytest cuml-dask (UCXX only)"
timeout -v 10m ./ci/run_cuml_dask_pytests.sh "${test_args[@]}" --run_ucx

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

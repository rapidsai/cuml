#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Support invoking test_python_singlegpu.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../ || exit 1

# Common setup steps shared by Python test jobs
source ./ci/test_python_common.sh

# We want to error if dask is installed in this environment so
if python -c 'import dask' 2>/dev/null; then
  echo "ERROR: dask is installed in this environment! This means we are not \
  testing that cuml works without dask. Please adjust the environment so the \
  main test environment doesn't install dask."
  exit 1
fi

# Install compute-sanitizer-api for the appropriate CUDA version
CUDA_VERSION_SHORT="${RAPIDS_CUDA_VERSION%.*}"
rapids-logger "Installing compute-sanitizer-api for CUDA ${CUDA_VERSION_SHORT}"
rapids-mamba-retry install -y -c nvidia compute-sanitizer-api cuda-version="${CUDA_VERSION_SHORT}"

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest cuml UMAP under compute-sanitizer"
cd python/cuml/tests || exit 1
compute-sanitizer --tool memcheck python -m pytest \
  --cache-clear \
  --numprocesses=8 \
  --dist=worksteal \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml-umap-sanitizer.xml" \
  --cov-config=../.coveragerc \
  --cov=cuml \
  --cov-report=xml:"${RAPIDS_COVERAGE_DIR}/cuml-umap-sanitizer-coverage.xml" \
  test_umap.py

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

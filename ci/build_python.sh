#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-generate-version > ./VERSION

rapids-logger "Begin py build"

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
LIBRAFT_CHANNEL=$(rapids-get-pr-conda-artifact raft 2530 cpp)
LIBCUVS_CHANNEL=$(rapids-get-pr-conda-artifact cuvs 540 cpp)
PYRAFT_CHANNEL=$(rapids-get-pr-conda-artifact raft 2530 python)

sccache --zero-stats

# TODO: Remove `--no-test` flag once importing on a CPU
# node works correctly
RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION) rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${LIBRAFT_CHANNEL}" \
  --channel "${LIBCUVS_CHANNEL}" \
  --channel "${PYRAFT_CHANNEL}" \
  conda/recipes/cuml

sccache --show-adv-stats

# Build cuml-cpu only in CUDA 11 jobs since it only depends on python
# version
RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
if [[ ${RAPIDS_CUDA_MAJOR} == "11" ]]; then
  sccache --zero-stats

  RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION) rapids-conda-retry mambabuild \
  --no-test \
  conda/recipes/cuml-cpu

  sccache --show-adv-stats
fi

rapids-upload-conda-to-s3 python

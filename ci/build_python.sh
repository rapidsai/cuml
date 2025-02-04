#!/bin/bash
# Copyright (c) 2022-2025, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-generate-version > ./VERSION

rapids-logger "Begin py build"

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
LIBRMM_CHANNEL=$(_rapids-get-pr-artifact rmm 1808 cpp conda)
LIBRAFT_CHANNEL=$(_rapids-get-pr-artifact raft 2566 cpp conda)
LIBCUVS_CHANNEL=$(_rapids-get-pr-artifact cuvs 644 cpp conda)
PYLIBRMM_CHANNEL=$(_rapids-get-pr-artifact rmm 1808 python conda)
PYLIBRAFT_CHANNEL=$(_rapids-get-pr-artifact libraft 2566 python conda)
PYLIBCUVS_CHANNEL=$(_rapids-get-pr-artifact cuvs 644 python conda)

sccache --zero-stats

# TODO: Remove `--no-test` flag once importing on a CPU
# node works correctly
RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION) rapids-conda-retry mambabuild \
  --no-test \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${LIBRAFT_CHANNEL}" \
  --channel "${LIBCUVS_CHANNEL}" \
  --channel "${PYLIBRMM_CHANNEL}" \
  --channel "${PYLIBRAFT_CHANNEL}" \
  --channel "${PYLIBCUVS_CHANNEL}" \
  --channel "${CPP_CHANNEL}" \
  conda/recipes/cuml

sccache --show-adv-stats

# Build cuml-cpu only in CUDA 11 jobs since it only depends on python
# version
RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
if [[ ${RAPIDS_CUDA_MAJOR} == "11" ]]; then
  sccache --zero-stats

  RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION) rapids-conda-retry mambabuild \
  --no-test \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${LIBRAFT_CHANNEL}" \
  --channel "${LIBCUVS_CHANNEL}" \
  --channel "${PYLIBRMM_CHANNEL}" \
  --channel "${PYLIBRAFT_CHANNEL}" \
  --channel "${PYLIBCUVS_CHANNEL}" \
  conda/recipes/cuml-cpu

  sccache --show-adv-stats
fi

rapids-upload-conda-to-s3 python

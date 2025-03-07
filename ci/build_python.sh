#!/bin/bash
# Copyright (c) 2022-2025, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

source rapids-telemetry-setup

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-generate-version > ./VERSION

rapids-logger "Begin py build"

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

sccache --zero-stats

# TODO: Remove `--no-test` flag once importing on a CPU
# node works correctly
RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION) rapids-conda-retry build \
  --no-test \
  --channel "${CPP_CHANNEL}" \
<<<<<<< HEAD
  conda/recipes/cuml 2>&1 | tee telemetry-artifacts/build.log

sccache --show-adv-stats | tee telemetry-artifacts/sccache-stats.txt
=======
  conda/recipes/cuml 2>&1 | tee ${GITHUB_WORKSPACE}/telemetry-artifacts/build.log

sccache --show-adv-stats | tee ${GITHUB_WORKSPACE}/telemetry-artifacts/sccache-stats.txt
>>>>>>> 8c29ddcdc12d827f81dd8c684c892922350513a2

# Build cuml-cpu only in CUDA 11 jobs since it only depends on python
# version
RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
if [[ ${RAPIDS_CUDA_MAJOR} == "11" ]]; then
  sccache --zero-stats

  RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION) rapids-conda-retry build \
  --no-test \
  conda/recipes/cuml-cpu

  sccache --show-adv-stats
fi

rapids-upload-conda-to-s3 python

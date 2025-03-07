#!/bin/bash
# Copyright (c) 2022-2025, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

source rapids-telemetry-setup

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin cpp build"

sccache --zero-stats

<<<<<<< HEAD
RAPIDS_PACKAGE_VERSION=$(rapids-generate-version) rapids-conda-retry mambabuild \
  conda/recipes/libcuml 2>&1 | tee telemetry-artifacts/build.log

sccache --show-adv-stats | tee telemetry-artifacts/sccache-stats.txt
=======
RAPIDS_PACKAGE_VERSION=$(rapids-generate-version) rapids-conda-retry build \
  conda/recipes/libcuml 2>&1 | tee ${GITHUB_WORKSPACE}/telemetry-artifacts/build.log

sccache --show-adv-stats | tee ${GITHUB_WORKSPACE}/telemetry-artifacts/sccache-stats.txt
>>>>>>> 8c29ddcdc12d827f81dd8c684c892922350513a2

rapids-upload-conda-to-s3 cpp

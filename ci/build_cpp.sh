#!/bin/bash
# Copyright (c) 2022-2025, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin cpp build"

sccache --zero-stats

RAPIDS_PACKAGE_VERSION=$(rapids-generate-version) rapids-conda-retry build \
  conda/recipes/libcuml 2>&1 | tee telemetry-artifacts/build.log

sccache --show-adv-stats | tee telemetry-artifacts/sccache-stats.txt

rapids-upload-conda-to-s3 cpp

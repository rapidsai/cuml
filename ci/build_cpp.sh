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
LIBRMM_CHANNEL=$(_rapids-get-pr-artifact rmm 1808 cpp conda)
LIBRAFT_CHANNEL=$(_rapids-get-pr-artifact raft 2566 cpp conda)
LIBCUVS_CHANNEL=$(_rapids-get-pr-artifact cuvs 644 cpp conda)

RAPIDS_PACKAGE_VERSION=$(rapids-generate-version) rapids-conda-retry mambabuild \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${LIBRAFT_CHANNEL}" \
  --channel "${LIBCUVS_CHANNEL}" \
  conda/recipes/libcuml

sccache --show-adv-stats

rapids-upload-conda-to-s3 cpp

#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin cpp build"

sccache --zero-stats

LIBRAFT_CHANNEL=$(rapids-get-pr-conda-artifact raft 2530 cpp)
LIBCUVS_CHANNEL=$(rapids-get-pr-conda-artifact cuvs 540 cpp)

RAPIDS_PACKAGE_VERSION=$(rapids-generate-version) rapids-conda-retry mambabuild \
        --channel "${LIBRAFT_CHANNEL}" \
        --channel "${LIBCUVS_CHANNEL}" \
        conda/recipes/libcuml

sccache --show-adv-stats

rapids-upload-conda-to-s3 cpp

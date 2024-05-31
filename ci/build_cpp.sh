#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

LIBRMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1544 cpp)
LIBRAFT_CHANNEL=$(rapids-get-pr-conda-artifact raft 2279 cpp)

version=$(rapids-generate-version)

rapids-logger "Begin cpp build"

RAPIDS_PACKAGE_VERSION=${version} rapids-conda-retry mambabuild \
    --channel "${LIBRMM_CHANNEL}" \
    --channel "${LIBRAFT_CHANNEL}" \
    conda/recipes/libcuml

rapids-upload-conda-to-s3 cpp

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

LIBRMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1776 cpp)
LIBRAFT_CHANNEL=$(rapids-get-pr-conda-artifact raft 2534 cpp)

RAPIDS_PACKAGE_VERSION=$(rapids-generate-version) rapids-conda-retry mambabuild --channel "${LIBRMM_CHANNEL}" --channel "${LIBRAFT_CHANNEL}" conda/recipes/libcuml

sccache --show-adv-stats

rapids-upload-conda-to-s3 cpp

#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin cpp build"

LIBRMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1678 cpp)
RMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1678 python)

UCXX_CHANNEL=$(rapids-get-pr-conda-artifact ucxx 278 cpp)

LIBCUDF_CHANNEL=$(rapids-get-pr-conda-artifact cudf 16806 cpp)
CUDF_CHANNEL=$(rapids-get-pr-conda-artifact cudf 16806 python)

LIBRAFT_CHANNEL=$(rapids-get-pr-conda-artifact raft 2433 cpp)
RAFT_CHANNEL=$(rapids-get-pr-conda-artifact raft 2433 python)

RAPIDS_PACKAGE_VERSION=$(rapids-generate-version) rapids-conda-retry mambabuild \
    --channel "${LIBRMM_CHANNEL}" \
    --channel "${RMM_CHANNEL}" \
    --channel "${UCXX_CHANNEL}" \
    --channel "${LIBCUDF_CHANNEL}" \
    --channel "${CUDF_CHANNEL}" \
    --channel "${LIBRAFT_CHANNEL}" \
    --channel "${RAFT_CHANNEL}" \
    conda/recipes/libcuml

rapids-upload-conda-to-s3 cpp

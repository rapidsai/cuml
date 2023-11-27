#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

LIBRMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1374 cpp)
LIBRAFT_CHANNEL=$(rapids-get-pr-conda-artifact raft 1957 cpp)
LIBCUMLPRIMS_MG_CHANNEL=$(rapids-get-pr-conda-artifact cumlprims_mg 160 cpp)

version=$(rapids-generate-version)

rapids-logger "Begin cpp build"

RAPIDS_PACKAGE_VERSION=${version} rapids-conda-retry mambabuild \
    --channel "${LIBRMM_CHANNEL}" \
    --channel "${LIBRAFT_CHANNEL}" \
    --channel "${LIBCUMLPRIMS_MG_CHANNEL}" \
    conda/recipes/libcuml

rapids-upload-conda-to-s3 cpp

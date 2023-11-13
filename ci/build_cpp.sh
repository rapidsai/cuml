#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

LIBRMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1095 cpp)
LIBCUDF_CHANNEL=$(rapids-get-pr-conda-artifact cudf 14365 cpp)
LIBRAFT_CHANNEL=$(rapids-get-pr-conda-artifact raft 1964 cpp)

rapids-print-env

version=$(rapids-generate-version)

rapids-logger "Begin cpp build"

RAPIDS_PACKAGE_VERSION=${version} rapids-conda-retry mambabuild \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${LIBCUDF_CHANNEL}" \
  --channel "${LIBRAFT_CHANNEL}" \
  conda/recipes/libcuml

rapids-upload-conda-to-s3 cpp

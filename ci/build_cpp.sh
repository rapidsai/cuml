#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

LIBCUDF_CHANNEL=$(rapids-get-artifact ci/cudf/pull-request/12587/046025a/cudf_conda_cpp_cuda11_$(arch).tar.gz)

rapids-print-env

rapids-logger "Begin cpp build"

rapids-mamba-retry mambabuild \
  --channel "${LIBCUDF_CHANNEL}" \
  conda/recipes/libcuml


rapids-upload-conda-to-s3 cpp

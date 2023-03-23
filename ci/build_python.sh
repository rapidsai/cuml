#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin py build"

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PY_VER=${RAPIDS_PY_VERSION//./}
LIBRAFT_CHANNEL=$(rapids-get-artifact ci/raft/pull-request/1365/a8a9bcd/raft_conda_cpp_cuda11_$(arch).tar.gz)
RAFT_CHANNEL=$(rapids-get-artifact ci/raft/pull-request/1365/a8a9bcd/raft_conda_python_cuda11_${PY_VER}_$(arch).tar.gz)


# TODO: Remove `--no-test` flag once importing on a CPU
# node works correctly
rapids-mamba-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${LIBRAFT_CHANNEL}" \
  --channel "${RAFT_CHANNEL}" \
  conda/recipes/cuml

rapids-upload-conda-to-s3 python

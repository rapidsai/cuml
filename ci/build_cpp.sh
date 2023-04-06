#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
LIBRMM_CHANNEL=$(rapids-get-artifact ci/rmm/pull-request/1223/72e0c74/rmm_conda_cpp_cuda${RAPIDS_CUDA_MAJOR}_$(arch).tar.gz)
LIBRAFT_CHANNEL=$(rapids-get-artifact ci/raft/pull-request/1388/7bddaee/raft_conda_cpp_cuda${RAPIDS_CUDA_MAJOR}_$(arch).tar.gz)
LIBCUMLPRIMS_CHANNEL=$(rapids-get-artifact ci/cumlprims_mg/pull-request/129/85effb7/cumlprims_mg_conda_cpp_cuda${RAPIDS_CUDA_MAJOR}_$(arch).tar.gz)

if [ "${RAPIDS_CUDA_MAJOR}" == 12 ]; then
cat << EOF > /opt/conda/.condarc
auto_update_conda: False
channels:
  - rapidsai
  - rapidsai-nightly
  - dask/label/dev
  - pytorch
  - nvidia
  - conda-forge
always_yes: true
number_channel_notices: 0
conda_build:
  set_build_id: false
  root_dir: /tmp/conda-bld-workspace
  output_folder: /tmp/conda-bld-output
EOF
fi

rapids-print-env

rapids-logger "Begin cpp build"

rapids-mamba-retry mambabuild --channel "${LIBRMM_CHANNEL}" --channel "${LIBRAFT_CHANNEL}" --channel "${LIBCUMLPRIMS_CHANNEL}" conda/recipes/libcuml

rapids-upload-conda-to-s3 cpp

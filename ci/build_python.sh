#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

# https://github.com/rapidsai/cuml/pull/5640/files

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

package_name="cuml"
package_dir="python"

version=$(rapids-generate-version)
git_commit=$(git rev-parse HEAD)
export RAPIDS_PACKAGE_VERSION=${version}

echo "${version}" > VERSION
sed -i "/^__git_commit__/ s/= .*/= \"${git_commit}\"/g" "${package_dir}/${package_name}/_version.py"

rapids-logger "Begin py build"

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
LIBRMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1544 cpp)
RMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1544 python)
LIBRAFT_CHANNEL=$(rapids-get-pr-conda-artifact raft 2279 cpp)
RAFT_CHANNEL=$(rapids-get-pr-conda-artifact raft 2279 python)
LIBCUDF_CHANNEL=$(rapids-get-pr-conda-artifact cudf 15603 cpp)
CUDF_CHANNEL=$(rapids-get-pr-conda-artifact cudf 15603 python)


# TODO: Remove `--no-test` flag once importing on a CPU
# node works correctly
rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${LIBRMM_CHANNEL}" \
  --channel "${RMM_CHANNEL}" \
  --channel "${LIBRAFT_CHANNEL}" \
  --channel "${RAFT_CHANNEL}" \
  --channel "${LIBCUDF_CHANNEL}" \
  --channel "${CUDF_CHANNEL}" \
  conda/recipes/cuml

# Build cuml-cpu only in CUDA 11 jobs since it only depends on python
# version
RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
if [[ ${RAPIDS_CUDA_MAJOR} == "11" ]]; then
  rapids-conda-retry mambabuild \
  --no-test \
  conda/recipes/cuml-cpu
fi

rapids-upload-conda-to-s3 python

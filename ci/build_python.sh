#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

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

REPO="rmm"
PR_NUMBER="1095"
COMMIT=$(git ls-remote https://github.com/rapidsai/${REPO}.git refs/heads/pull-request/${PR_NUMBER} | cut -c1-7)
RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
PYTHON_MINOR_VERSION=$(python --version | sed -E 's/Python [0-9]+\.([0-9]+)\.[0-9]+/\1/g')
LIBRMM_CHANNEL=$(rapids-get-artifact ci/${REPO}/pull-request/${PR_NUMBER}/${COMMIT}/rmm_conda_cpp_cuda${RAPIDS_CUDA_MAJOR}_$(arch).tar.gz)
RMM_CHANNEL=$(rapids-get-artifact ci/${REPO}/pull-request/${PR_NUMBER}/${COMMIT}/rmm_conda_python_cuda${RAPIDS_CUDA_MAJOR}_3${PYTHON_MINOR_VERSION}_$(arch).tar.gz)

# TODO: Remove `--no-test` flag once importing on a CPU
# node works correctly
rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  --channel "${RMM_CHANNEL}" \
  --channel "${LIBRMM_CHANNEL}" \
  conda/recipes/cuml

# Build cuml-cpu only in CUDA 11 jobs since it only depends on python
# version
RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
if [[ ${RAPIDS_CUDA_MAJOR} == "11" ]]; then
  rapids-conda-retry mambabuild \
  --no-test \
  --channel "${RMM_CHANNEL}" \
  --channel "${LIBRMM_CHANNEL}" \
  conda/recipes/cuml-cpu
fi

rapids-upload-conda-to-s3 python

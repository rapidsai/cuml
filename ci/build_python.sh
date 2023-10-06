#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

package_name="cuml"
package_dir="python"

version=$(rapids-generate-version)
commit_override=$(git rev-parse HEAD)

sed -i "s/__version__ = .*/__version__ = ${version}/g" ${package_dir}/${package_name}/__init__.py
sed -i "s/__git_commit__ = .*/__git_commit__ = \"${commit_override}\"/g" ${package_dir}/${package_name}/__init__.py

rapids-logger "Begin py build"

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

# TODO: Remove `--no-test` flag once importing on a CPU
# node works correctly
RAPIDS_PACKAGE_VERSION=${version} rapids-conda-retry mambabuild \
  --no-test \
  --channel "${CPP_CHANNEL}" \
  conda/recipes/cuml

# Build cuml-cpu only in CUDA 11 jobs since it only depends on python
# version
RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
if [[ ${RAPIDS_CUDA_MAJOR} == "11" ]]; then
  RAPIDS_PACKAGE_VERSION=${version} rapids-conda-retry mambabuild \
  --no-test \
  conda/recipes/cuml-cpu
fi

rapids-upload-conda-to-s3 python

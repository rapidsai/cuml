#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_name="cuml"
package_dir="python"

source rapids-configure-sccache
source rapids-date-string

version=$(rapids-generate-version)

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# This is the version of the suffix with a preceding hyphen. It's used
# everywhere except in the final wheel name.
PACKAGE_CUDA_SUFFIX="-${RAPIDS_PY_CUDA_SUFFIX}"

# Patch project metadata files to include the CUDA version suffix and version override.
pyproject_file="${package_dir}/pyproject.toml"

echo "${version}" > VERSION

if [[ $PACKAGE_CUDA_SUFFIX == "-cu12" ]]; then
    sed -i "s/cuda-python[<=>\.,0-9]*/cuda-python>=12.0,<13.0/g" ${pyproject_file}
fi

cd ${package_dir}

# For testing, will be removed when the package is released
git clone -b use-dfg https://github.com/KyleFromNVIDIA/rapids-build-backend.git
pushd rapids-build-backend
python -m pip wheel . --no-deps
popd

PIP_FIND_LINKS="$PWD/rapids-build-backend" SKBUILD_CMAKE_ARGS="-DDETECT_CONDA_ENV=OFF;-DDISABLE_DEPRECATION_WARNINGS=ON;-DCPM_cumlprims_mg_SOURCE=${GITHUB_WORKSPACE}/cumlprims_mg/" \
  python -m pip wheel . \
    -w dist \
    -vvv \
    --no-deps \
    --disable-pip-version-check

mkdir -p final_dist
python -m auditwheel repair -w final_dist dist/*

RAPIDS_PY_WHEEL_NAME="cuml_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 final_dist

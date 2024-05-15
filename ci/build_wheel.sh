#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

package_name="cuml"
package_dir="python"

source rapids-configure-sccache
source rapids-date-string

version=$(rapids-generate-version)
git_commit=$(git rev-parse HEAD)

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# This is the version of the suffix with a preceding hyphen. It's used
# everywhere except in the final wheel name.
PACKAGE_CUDA_SUFFIX="-${RAPIDS_PY_CUDA_SUFFIX}"

# Patch project metadata files to include the CUDA version suffix and version override.
pyproject_file="${package_dir}/pyproject.toml"

sed -i "/^name.*cuml/ s/= \"cuml\"/= \"cuml${PACKAGE_CUDA_SUFFIX}\"/g" ${pyproject_file}
echo "${version}" > VERSION
sed -i "/^__git_commit__ / s/= .*/= \"${git_commit}\"/g" "${package_dir}/${package_name}/_version.py"

# For nightlies we want to ensure that we're pulling in alphas as well. The
# easiest way to do so is to augment the spec with a constraint containing a
# min alpha version that doesn't affect the version bounds but does allow usage
# of alpha versions for that dependency without --pre
alpha_spec=''
if ! rapids-is-release-build; then
    alpha_spec=',>=0.0.0a0'
fi

for dep in cudf pylibraft raft-dask rmm; do
    sed -r -i "s/${dep}==(.*)\"/${dep}${PACKAGE_CUDA_SUFFIX}==\1${alpha_spec}\"/g" ${pyproject_file}
done

for dep in dask-cuda rapids-dask-dependency; do
    sed -r -i "s/${dep}==(.*)\"/${dep}==\1${alpha_spec}\"/g" ${pyproject_file}
done

if [[ $PACKAGE_CUDA_SUFFIX == "-cu12" ]]; then
    sed -i "s/cuda-python[<=>\.,0-9]*/cuda-python>=12.0,<13.0/g" ${pyproject_file}
    sed -i "s/cupy-cuda11x/cupy-cuda12x/g" ${pyproject_file}
fi

cd ${package_dir}

SKBUILD_CMAKE_ARGS="-DDETECT_CONDA_ENV=OFF;-DDISABLE_DEPRECATION_WARNINGS=ON;-DCPM_cumlprims_mg_SOURCE=${GITHUB_WORKSPACE}/cumlprims_mg/" \
  python -m pip wheel . \
    -w dist \
    -vvv \
    --no-deps \
    --disable-pip-version-check

mkdir -p final_dist
python -m auditwheel repair -w final_dist dist/*

RAPIDS_PY_WHEEL_NAME="cuml_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 final_dist

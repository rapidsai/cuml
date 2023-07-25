#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-configure-sccache
source rapids-date-string

# Use gha-tools rapids-pip-wheel-version to generate wheel version then
# update the necessary files
version_override="$(rapids-pip-wheel-version ${RAPIDS_DATE_STRING})"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

ci/release/apply_wheel_modifications.sh ${version_override} "-${RAPIDS_PY_CUDA_SUFFIX}"
echo "The package name and/or version was modified in the package source. The git diff is:"
git diff

cd python/

SKBUILD_CONFIGURE_OPTIONS="-DCUML_BUILD_WHEELS=ON -DDETECT_CONDA_ENV=OFF -DDISABLE_DEPRECATION_WARNINGS=ON -DCPM_cumlprims_mg_SOURCE=${GITHUB_WORKSPACE}/cumlprims_mg/" \
  python -m pip wheel . \
    -w dist \
    -vvv \
    --no-deps \
    --disable-pip-version-check

mkdir -p final_dist
python -m auditwheel repair -w final_dist dist/*

RAPIDS_PY_WHEEL_NAME="cuml_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 final_dist

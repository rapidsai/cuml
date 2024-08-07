#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/cuml"

source rapids-configure-sccache
source rapids-date-string

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# This is the version of the suffix with a preceding hyphen. It's used
# everywhere except in the final wheel name.
PACKAGE_CUDA_SUFFIX="-${RAPIDS_PY_CUDA_SUFFIX}"

rapids-generate-version > ./VERSION


# Downloads libcuml wheel from this current build,
# then ensures 'cuml' wheel builds always use the 'libcuml' just built in the same CI run.
#
# Using env variable PIP_CONSTRAINT is necessary to ensure the constraints
# are used when creating the isolated build environment.
RAPIDS_PY_WHEEL_NAME="libcuml_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp /tmp/libcuml_dist

echo "libcuml-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo /tmp/libcuml_dist/libcuml_*.whl)" > /tmp/constraints.txt
export PIP_CONSTRAINT="/tmp/constraints.txt"

cd ${package_dir}

SKBUILD_CMAKE_ARGS="-DDETECT_CONDA_ENV=OFF;-DDISABLE_DEPRECATION_WARNINGS=ON;-DCPM_cumlprims_mg_SOURCE=${GITHUB_WORKSPACE}/cumlprims_mg/" \
  python -m pip wheel . \
    -w dist \
    -vvv \
    --no-deps \
    --disable-pip-version-check

mkdir -p final_dist
python -m auditwheel repair -w final_dist dist/*

RAPIDS_PY_WHEEL_NAME="cuml_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 python final_dist

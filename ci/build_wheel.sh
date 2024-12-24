#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

package_dir="python/cuml"

source rapids-configure-sccache
source rapids-date-string

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

rapids-generate-version > ./VERSION

cd ${package_dir}

case "${RAPIDS_CUDA_VERSION}" in
  12.*)
    EXCLUDE_ARGS=(
      --exclude "libcuvs.so"
      --exclude "libcublas.so.12"
      --exclude "libcublasLt.so.12"
      --exclude "libcufft.so.11"
      --exclude "libcurand.so.10"
      --exclude "libcusolver.so.11"
      --exclude "libcusparse.so.12"
      --exclude "libnvJitLink.so.12"
    )
    EXTRA_CMAKE_ARGS=";-DUSE_CUDA_MATH_WHEELS=ON"
    ;;
  11.*)
    EXCLUDE_ARGS=(
      --exclude "libcuvs.so"
    )
    EXTRA_CMAKE_ARGS=";-DUSE_CUDA_MATH_WHEELS=OFF"
    ;;
esac

sccache --zero-stats

SKBUILD_CMAKE_ARGS="-DDETECT_CONDA_ENV=OFF;-DDISABLE_DEPRECATION_WARNINGS=ON;-DCPM_cumlprims_mg_SOURCE=${GITHUB_WORKSPACE}/cumlprims_mg/;-DUSE_CUVS_WHEEL=ON${EXTRA_CMAKE_ARGS}" \
  python -m pip wheel . \
    -w dist \
    -v \
    --no-deps \
    --disable-pip-version-check

sccache --show-adv-stats

mkdir -p final_dist
python -m auditwheel repair -w final_dist "${EXCLUDE_ARGS[@]}" dist/*

../../ci/validate_wheel.sh final_dist

RAPIDS_PY_WHEEL_NAME="cuml_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 python final_dist

#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

package_name="cuml"
package_dir="python/cuml"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# TODO(jameslamb): split this out into build_wheel_{cuml,libcuml}.sh
# TODO(jameslamb): add libcuml++.so to cuml exclusions
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

export SKBUILD_CMAKE_ARGS="-DDETECT_CONDA_ENV=OFF;-DDISABLE_DEPRECATION_WARNINGS=ON;-DCPM_cumlprims_mg_SOURCE=${GITHUB_WORKSPACE}/cumlprims_mg/;-DUSE_CUVS_WHEEL=ON${EXTRA_CMAKE_ARGS}"
./ci/build_wheel.sh "${package_name}" "${package_dir}"

mkdir -p ${package_dir}/final_dist
python -m auditwheel repair \
    "${EXCLUDE_ARGS[@]}" \
    -w ${package_dir}/final_dist \
    ${package_dir}/dist/*

./ci/validate_wheel.sh ${package_dir} final_dist

RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 python "${package_dir}/final_dist"

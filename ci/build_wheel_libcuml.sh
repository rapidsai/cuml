#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

set -euo pipefail

package_name="libcuml"
package_dir="python/libcuml"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

rapids-logger "Generating build requirements"

rapids-dependency-file-generator \
  --output requirements \
  --file-key "py_build_${package_name}" \
  --file-key "py_rapids_build_${package_name}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};cuda_suffixed=true" \
| tee /tmp/requirements-build.txt

rapids-logger "Installing build requirements"
python -m pip install \
    -v \
    --prefer-binary \
    -r /tmp/requirements-build.txt

# build with '--no-build-isolation', for better sccache hit rate
# 0 really means "add --no-build-isolation" (ref: https://github.com/pypa/pip/issues/5735)
export PIP_NO_BUILD_ISOLATION=0

# NOTE: 'libcumlprims_mg.so' is marked as '--exclude' here because auditwheel doesn't detect it,
#       but it really is intentionally included in 'libcuml' wheels
EXCLUDE_ARGS=(
  --exclude "libcumlprims_mg.so"
  --exclude "libcuvs.so"
  --exclude "libraft.so"
)

case "${RAPIDS_CUDA_VERSION}" in
  12.*)
    EXCLUDE_ARGS+=(
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
    EXTRA_CMAKE_ARGS=";-DUSE_CUDA_MATH_WHEELS=OFF"
    ;;
esac

export SKBUILD_CMAKE_ARGS="-DDISABLE_DEPRECATION_WARNINGS=ON;-DCPM_cumlprims_mg_SOURCE=${GITHUB_WORKSPACE}/cumlprims_mg/;-DUSE_CUVS_WHEEL=ON${EXTRA_CMAKE_ARGS};-DSINGLEGPU=OFF"
./ci/build_wheel.sh "${package_name}" "${package_dir}"

mkdir -p ${package_dir}/final_dist
python -m auditwheel repair \
    "${EXCLUDE_ARGS[@]}" \
    -w ${package_dir}/final_dist \
    ${package_dir}/dist/*

./ci/validate_wheel.sh ${package_dir} final_dist

RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 cpp "${package_dir}/final_dist"

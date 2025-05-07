#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

package_name="cuml"
package_dir="python/cuml"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"

# Download the libcuml wheel built in the previous step and make it
# available for pip to find.
LIBCUML_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcuml_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
echo "libcuml-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo "${LIBCUML_WHEELHOUSE}"/libcuml_*.whl)" >> /tmp/constraints.txt
export PIP_CONSTRAINT="/tmp/constraints.txt"

EXCLUDE_ARGS=(
  --exclude "libcuml++.so"
  --exclude "libcumlprims_mg.so"
  --exclude "libcuvs.so"
  --exclude "libraft.so"
  --exclude "libcublas.so.*"
  --exclude "libcublasLt.so.*"
  --exclude "libcufft.so.*"
  --exclude "libcurand.so.*"
  --exclude "libcusolver.so.*"
  --exclude "libcusparse.so.*"
  --exclude "libnvJitLink.so.*"
  --exclude "librapids_logger.so"
  --exclude "librmm.so"
)

export SKBUILD_CMAKE_ARGS="-DDISABLE_DEPRECATION_WARNINGS=ON;-DSINGLEGPU=OFF;-DUSE_LIBCUML_WHEEL=ON"
./ci/build_wheel.sh "${package_name}" "${package_dir}"

# repair wheels and write to the location that artifact-uploading code expects to find them
python -m auditwheel repair \
    "${EXCLUDE_ARGS[@]}" \
    -w "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}" \
    ${package_dir}/dist/*

./ci/validate_wheel.sh ${package_dir} "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 python "${RAPIDS_WHEEL_BLD_OUTPUT_DIR}"

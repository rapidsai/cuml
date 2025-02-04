#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

package_name="cuml"
package_dir="python/cuml"

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"

# Download the libcuml wheel built in the previous step and make it
# available for pip to find.
RAPIDS_PY_WHEEL_NAME="libcuml_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp /tmp/libcuml_dist
echo "libcuml-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo /tmp/libcuml_dist/libcuml_*.whl)" >> /tmp/constraints.txt

LIBRMM_WHEEL_DIR=$(RAPIDS_PY_WHEEL_NAME="rmm_${RAPIDS_PY_CUDA_SUFFIX}" _rapids-get-pr-artifact rmm 1808 cpp wheel)
PYLIBRMM_WHEEL_DIR=$(RAPIDS_PY_WHEEL_NAME="rmm_${RAPIDS_PY_CUDA_SUFFIX}" _rapids-get-pr-artifact rmm 1808 python wheel)
LIBRAFT_WHEEL_DIR=$(RAPIDS_PY_WHEEL_NAME="libraft_${RAPIDS_PY_CUDA_SUFFIX}" _rapids-get-pr-artifact raft 2566 cpp wheel)
PYLIBRAFT_WHEEL_DIR=$(RAPIDS_PY_WHEEL_NAME="pylibraft_${RAPIDS_PY_CUDA_SUFFIX}" _rapids-get-pr-artifact raft 2566 python wheel)
LIBCUVS_WHEEL_DIR=$(RAPIDS_PY_WHEEL_NAME="libcuvs_${RAPIDS_PY_CUDA_SUFFIX}" _rapids-get-pr-artifact cuvs 644 cpp wheel)
PYLIBCUVS_WHEEL_DIR=$(RAPIDS_PY_WHEEL_NAME="cuvs_${RAPIDS_PY_CUDA_SUFFIX}" _rapids-get-pr-artifact cuvs 644 python wheel)
echo "librmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBRMM_WHEEL_DIR}/librmm_*.whl)" >> /tmp/constraints.txt
echo "rmm-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${PYLIBRMM_WHEEL_DIR}/rmm_*.whl)" >> /tmp/constraints.txt
echo "libraft-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBRAFT_WHEEL_DIR}/libraft_*.whl)" >> /tmp/constraints.txt
echo "raft-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${PYLIBRAFT_WHEEL_DIR}/raft_*.whl)" >> /tmp/constraints.txt
echo "libcuvs-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${LIBRAFT_WHEEL_DIR}/libcuvs_*.whl)" >> /tmp/constraints.txt
echo "cuvs-${RAPIDS_PY_CUDA_SUFFIX} @ file://$(echo ${PYLIBRAFT_WHEEL_DIR}/cuvs_*.whl)" >> /tmp/constraints.txt

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
)

export SKBUILD_CMAKE_ARGS="-DDISABLE_DEPRECATION_WARNINGS=ON;-DSINGLEGPU=OFF;-DUSE_LIBCUML_WHEEL=ON"
./ci/build_wheel.sh "${package_name}" "${package_dir}"

mkdir -p ${package_dir}/final_dist
python -m auditwheel repair \
    "${EXCLUDE_ARGS[@]}" \
    -w ${package_dir}/final_dist \
    ${package_dir}/dist/*

./ci/validate_wheel.sh ${package_dir} final_dist

RAPIDS_PY_WHEEL_NAME="${package_name}_${RAPIDS_PY_CUDA_SUFFIX}" rapids-upload-wheels-to-s3 python "${package_dir}/final_dist"

#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

mkdir -p ./dist
RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen ${RAPIDS_CUDA_VERSION})"
RAPIDS_PY_WHEEL_NAME="cuml_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 python ./dist
RAPIDS_PY_WHEEL_NAME="libcuml_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-s3 cpp ./dist

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

# echo to expand wildcard before adding `[extra]` requires for pip
python -m pip install \
  ./dist/libcuml*.whl \
  "$(echo ./dist/cuml*.whl)[test]"

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
mkdir -p "${RAPIDS_TESTS_DIR}"

EXITCODE=0
trap "EXITCODE=1" ERR
set +e


rapids-logger "pytest cuml single GPU"
./ci/run_cuml_singlegpu_pytests.sh \
  --numprocesses=8 \
  --dist=worksteal \
  -k 'not test_sparse_pca_inputs' \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml.xml"

# Run test_sparse_pca_inputs separately
./ci/run_cuml_singlegpu_pytests.sh \
  -k 'test_sparse_pca_inputs' \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml-sparse-pca.xml"

rapids-logger "pytest cuml-dask"
./ci/run_cuml_dask_pytests.sh \
  --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml-dask.xml"

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

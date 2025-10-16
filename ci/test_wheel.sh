#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
CUML_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cuml_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)
LIBCUML_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcuml_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
mkdir -p "${RAPIDS_TESTS_DIR}"

# generate constraints, the constraints will limit the version of the
# dependencies that can be installed later on when installing the wheel
rapids-generate-pip-constraints test_python ./constraints.txt

# Install just minimal dependencies first
rapids-pip-retry install \
  "${LIBCUML_WHEELHOUSE}"/libcuml*.whl \
  "${CUML_WHEELHOUSE}"/cuml*.whl \
  --constraint ./constraints.txt \
  --constraint "${PIP_CONSTRAINT}"

# Try to import cuml with just a minimal install"
rapids-logger "Importing cuml with minimal dependencies"
python -c "import cuml"

# notes:
#
#   * echo to expand wildcard before adding `[test,experimental]` requires for pip
#   * need to provide --constraint="${PIP_CONSTRAINT}" because that environment variable is
#     ignored if any other --constraint are passed via the CLI
#
rapids-pip-retry install \
   "${LIBCUML_WHEELHOUSE}"/libcuml*.whl \
  "$(echo "${CUML_WHEELHOUSE}"/cuml*.whl)[test]" \
  --constraint ./constraints.txt \
  --constraint "${PIP_CONSTRAINT}"

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Run linkage test for libcuml
rapids-logger "Testing libcuml linkage"
python -m pytest --cache-clear python/libcuml/tests/test_libcuml_linkage.py -v

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

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

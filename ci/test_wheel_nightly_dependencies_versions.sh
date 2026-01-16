#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Test cuml with the nightly versions of its dependencies.

set -euo pipefail

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
CUML_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cuml_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)
LIBCUML_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcuml_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
mkdir -p "${RAPIDS_TESTS_DIR}"

# Explicitly install the packages (and their dependencies) that we want to have from the nightly index.
rapids-pip-retry install \
  --pre \
  --extra-index-url=https://pypi.anaconda.org/scientific-python-nightly-wheels/simple \
  scikit-learn

rapids-dependency-file-generator \
  --output requirements \
  --file-key test_python \
  --matrix "dependencies=latest" | tee requirements.txt

# notes:
#
#   * echo to expand wildcard before adding `[test,experimental]` requires for pip
#   * need to provide --constraint="${PIP_CONSTRAINT}" because that environment variable is
#     ignored if any other --constraint are passed via the CLI
#
rapids-pip-retry install \
   "${LIBCUML_WHEELHOUSE}"/libcuml*.whl \
  "$(echo "${CUML_WHEELHOUSE}"/cuml*.whl)[test]" \
  --requirement requirements.txt \
  --constraint "${PIP_CONSTRAINT}"

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest cuml single GPU"
timeout 1h ./ci/run_cuml_singlegpu_pytests.sh \
  --numprocesses=8 \
  --dist=worksteal

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

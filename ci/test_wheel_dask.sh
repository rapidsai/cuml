#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
CUML_WHEELHOUSE=$(rapids-download-from-github "$(rapids-package-name "wheel_python" cuml --stable --cuda "$RAPIDS_CUDA_VERSION")")
LIBCUML_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcuml_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
mkdir -p "${RAPIDS_TESTS_DIR}"

# generate constraints, the constraints will limit the version of the
# dependencies that can be installed later on when installing the wheel
rapids-generate-pip-constraints test_python ./constraints.txt

# notes:
#
#   * echo to expand wildcard before adding `[...]` extras for pip
#   * need to provide --constraint="${PIP_CONSTRAINT}" because that environment variable is
#     ignored if any other --constraint are passed via the CLI
#
rapids-pip-retry install \
   "${LIBCUML_WHEELHOUSE}"/libcuml*.whl \
  "$(echo "${CUML_WHEELHOUSE}"/cuml*.whl)[dask,test,test-dask]" \
  --constraint ./constraints.txt \
  --constraint "${PIP_CONSTRAINT}"

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

rapids-logger "pytest cuml dask"
timeout 1h ./ci/run_cuml_dask_pytests.sh --junitxml="${RAPIDS_TESTS_DIR}/junit-cuml-dask.xml"

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

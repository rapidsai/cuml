#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-github cpp)
PYTHON_CHANNEL=$(rapids-download-from-github "$(rapids-package-name "conda_python" cuml --stable --cuda "${RAPIDS_CUDA_VERSION}")")

rapids-logger "Generate Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key "${DEPENDENCY_FILE_KEY:-test_python}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};dependencies=${RAPIDS_DEPENDENCIES}" \
  --prepend-channel "${CPP_CHANNEL}" \
  --prepend-channel "${PYTHON_CHANNEL}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

# When libcuml is built against a pinned RAFT fork (CPM), reinstall pylibraft from
# the same fork/tag so Python raft::handle_t matches libcuml (avoids segfaults in
# get_resource / stream pool during pytest collection).
GET_RAFT_CMAKE="cpp/cmake/thirdparty/get_raft.cmake"
if [[ -f "${GET_RAFT_CMAKE}" ]]; then
  RAFT_FORK=$(sed -n 's/set(RAFT_FORK "\([^"]*\)".*/\1/p' "${GET_RAFT_CMAKE}" | head -1)
  RAFT_PINNED_TAG=$(sed -n 's/set(RAFT_PINNED_TAG "\([^"]*\)".*/\1/p' "${GET_RAFT_CMAKE}" | head -1)
  if [[ -n "${RAFT_FORK}" && -n "${RAFT_PINNED_TAG}" && "${RAFT_FORK}" != "rapidsai" ]]; then
    rapids-logger "Reinstalling pylibraft from ${RAFT_FORK}/raft@${RAFT_PINNED_TAG} (matches libcuml CPM RAFT pin)"
    python -m pip install --force-reinstall --no-deps \
      "pylibraft @ git+https://github.com/${RAFT_FORK}/raft.git@${RAFT_PINNED_TAG}#subdirectory=python/pylibraft"
  fi
fi

# dask and other tests sporadically run into this issue in ARM tests
# exception=ImportError('/opt/conda/envs/test/lib/python3.10/site-packages/cuml/internals/../../../.././libgomp.so.1: cannot allocate memory in static TLS block')>)
# this should avoid that/opt/conda/lib
if [[ "$(arch)" == "aarch64" ]]; then
  export LD_PRELOAD=/opt/conda/envs/test/lib/libgomp.so.1
fi

rapids-print-env

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
mkdir -p "${RAPIDS_TESTS_DIR}"

rapids-logger "Check GPU usage"
nvidia-smi

# Enable hypothesis testing for nightly test runs.
if [ "${RAPIDS_BUILD_TYPE}" == "nightly" ]; then
  export HYPOTHESIS_ENABLED="true"
fi

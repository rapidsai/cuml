#!/bin/bash
# Copyright (c) 2022-2025, NVIDIA CORPORATION.

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)
LIBRMM_CHANNEL=$(_rapids-get-pr-artifact rmm 1808 cpp conda)
LIBRAFT_CHANNEL=$(_rapids-get-pr-artifact raft 2566 cpp conda)
LIBCUVS_CHANNEL=$(_rapids-get-pr-artifact cuvs 644 cpp conda)
PYLIBRMM_CHANNEL=$(_rapids-get-pr-artifact rmm 1808 python conda)
PYLIBRAFT_CHANNEL=$(_rapids-get-pr-artifact raft 2566 python conda)
PYLIBCUVS_CHANNEL=$(_rapids-get-pr-artifact cuvs 644 python conda)

rapids-logger "Generate Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_python \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
  --prepend-channel "${CPP_CHANNEL}" \
  --prepend-channel "${LIBRMM_CHANNEL}" \
  --prepend-channel "${LIBRAFT_CHANNEL}" \
  --prepend-channel "${LIBCUVS_CHANNEL}" \
  --prepend-channel "${PYLIBRMM_CHANNEL}" \
  --prepend-channel "${PYLIBRAFT_CHANNEL}" \
  --prepend-channel "${PYLIBCUVS_CHANNEL}" \
  --prepend-channel "${PYTHON_CHANNEL}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

# dask and other tests sporadically run into this issue in ARM tests
# exception=ImportError('/opt/conda/envs/test/lib/python3.10/site-packages/cuml/internals/../../../.././libgomp.so.1: cannot allocate memory in static TLS block')>)
# this should avoid that/opt/conda/lib
if [[ "$(arch)" == "aarch64" ]]; then
  export LD_PRELOAD=/opt/conda/envs/test/lib/libgomp.so.1
fi

rapids-print-env

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"

rapids-logger "Check GPU usage"
nvidia-smi

# Enable hypothesis testing for nightly test runs.
if [ "${RAPIDS_BUILD_TYPE}" == "nightly" ]; then
  export HYPOTHESIS_ENABLED="true"
fi

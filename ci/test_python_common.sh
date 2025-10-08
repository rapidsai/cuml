#!/bin/bash
# Copyright (c) 2022-2025, NVIDIA CORPORATION.

set -euo pipefail

source ./ci/use_conda_packages_from_prs.sh

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-github cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-github python)

rapids-logger "Compressing repodata.json → repodata.json.zst for local channels"
for ch in "${CUVS_CHANNEL:-}" "${LIBCUVS_CHANNEL:-}" "${CPP_CHANNEL:-}"; do
  if [ -n "$ch" ] && [ -d "$ch" ]; then
    for subdir in "$ch"/linux-aarch64 "$ch"/linux-64 "$ch"/noarch; do
      if [ -f "${subdir}/repodata.json" ] && [ ! -f "${subdir}/repodata.json.zst" ]; then
        echo "Compressing ${subdir}/repodata.json → ${subdir}/repodata.json.zst"
        zstd -q -19 "${subdir}/repodata.json" -o "${subdir}/repodata.json.zst"
      fi
    done
  fi
done

echo "=== DEBUG: Listing contents of downloaded conda channels ==="
for ch in "${CUVS_CHANNEL:-}" "${LIBCUVS_CHANNEL:-}" "${CPP_CHANNEL:-}"; do
  if [ -n "$ch" ] && [ -d "$ch" ]; then
    echo ">>> $ch"
    find "$ch" -maxdepth 2 -type f | sort
  else
    echo "!!! Channel path $ch does not exist or is empty"
  fi
done
echo "============================================================="

rapids-logger "Generate Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key "${DEPENDENCY_FILE_KEY:-test_python}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION};dependencies=${RAPIDS_DEPENDENCIES}" \
  --prepend-channel "${LIBCUVS_CHANNEL}" \
  --prepend-channel "${CUVS_CHANNEL}" \
  --prepend-channel "${CPP_CHANNEL}" \
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

#!/bin/bash
# Copyright (c) 2022-2025, NVIDIA CORPORATION.

set -euo pipefail

source ./ci/use_conda_packages_from_prs.sh

# Support invoking test_cpp.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-github cpp)

rapids-logger "Compressing repodata.json → repodata.json.zst for local channels"
for ch in "${CUVS_CHANNEL:-}" "${LIBCUVS_CHANNEL:-}" "${CPP_CHANNEL:-}"; do
  if [ -n "$ch" ] && [ -d "$ch" ]; then
    for subdir in "$ch"/linux-64 "$ch"/noarch; do
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

rapids-logger "Generate C++ testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_cpp \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" \
  --prepend-channel "${CPP_CHANNEL}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-print-env

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

rapids-logger "Check GPU usage"
nvidia-smi

rapids-logger "Run gtests"
export GTEST_OUTPUT=xml:${RAPIDS_TESTS_DIR}/
# Run libcuml gtests from libcuml-tests package
./ci/run_ctests.sh -j9 && EXITCODE=$? || EXITCODE=$?;

rapids-logger "Test script exiting with value: $EXITCODE"
exit "${EXITCODE}"

#!/bin/bash

set -euo pipefail

# TODO: Remove
. /opt/conda/etc/profile.d/conda.sh
conda activate base

# Check environment
source ci/check_env.sh

gpuci_logger "Check GPU usage"
nvidia-smi

# GPU Test Stage
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)

# Install libcudf packages
gpuci_mamba_retry install \
  -c "${CPP_CHANNEL}" \
  libcuml libcuml-tests

# Run libcuml gtests from libcuml-tests package
TESTRESULTS_DIR=test-results
mkdir -p "${TESTRESULTS_DIR}"
SUITEERROR=0

gpuci_logger "Run cuml test"
set +e
for gt in "$CONDA_PREFIX/bin/gtests/libcuml/"* ; do
  echo "Running GoogleTest ${gt}"
  ${gt} --gtest_output=xml:"${TESTRESULTS_DIR}/libcuml_cpp/"
  EXITCODE=$?
    if (( EXITCODE != 0 )); then
        SUITEERROR="${EXITCODE}"
        echo "FAILED: GTest ${gt}"
    fi
done
set -e

gpuci_logger "Run ml-prims test"
set +e
for gt in "$CONDA_PREFIX/bin/gtests/libcuml_prims/"* ; do
  echo "Running GoogleTest ${gt}"
  ${gt} --gtest_output=xml:"${TESTRESULTS_DIR}/prims"
  EXITCODE=$?
    if (( EXITCODE != 0 )); then
        SUITEERROR="${EXITCODE}"
        echo "FAILED: GTest ${gt}"
    fi
done

exit "${SUITEERROR}"

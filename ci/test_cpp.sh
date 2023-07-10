#!/bin/bash
# Copyright (c) 2022-2023, NVIDIA CORPORATION.

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Generate C++ testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file_key test_cpp \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch)" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}/
mkdir -p "${RAPIDS_TESTS_DIR}"

rapids-print-env

# debug: remove cuda 12 packages from conda-forge that are being combined with nvidia channel ones
conda remove --force cuda-cccl_linux-64 cuda-cudart cuda-cudart-dev cuda-cudart-dev_linucuda-cudart-static  cuda-cudart-static_lincuda-profiler-api cuda-version libcublas libcublas-dev libcurand libcurand-dev libcusolver          libcusolver-dev libcusparse libcusparse-dev libnvjitlink

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  libcuml libcuml-tests

conda list

rapids-logger "Check GPU usage"
nvidia-smi

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

# Run libcuml gtests from libcuml-tests package
rapids-logger "Run gtests"
for gt in "$CONDA_PREFIX"/bin/gtests/libcuml/* ; do
    test_name=$(basename ${gt})
    echo "Running gtest $test_name"
    ${gt} --gtest_output=xml:${RAPIDS_TESTS_DIR}
done

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

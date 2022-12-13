#!/bin/bash
# Copyright (c) 2022, NVIDIA CORPORATION.

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
SUITEERROR=0

rapids-print-env

rapids-mamba-retry install \
  -c "${CPP_CHANNEL}" \
  libcuml libcuml-tests

rapids-logger "Check GPU usage"
nvidia-smi

set +e

# Run libcuml gtests from libcuml-tests package
rapids-logger "Run gtests"

# TODO: exit code handling is too verbose. Find a cleaner solution.

for gt in "$CONDA_PREFIX"/bin/gtests/libcuml/* ; do
    test_name=$(basename ${gt})
    echo "Running gtest $test_name"
    ${gt} --gtest_output=xml:${RAPIDS_TESTS_DIR}

    exitcode=$?
    if (( ${exitcode} != 0 )); then
        SUITEERROR=${exitcode}
        echo "FAILED: GTest ${gt}"
    fi
done

exit ${SUITEERROR}

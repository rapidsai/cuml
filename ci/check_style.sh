#!/bin/bash
# Copyright (c) 2020-2023, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Create checks conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
  --output conda \
  --file_key checks \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n checks
conda activate checks

# Run pre-commit checks
pre-commit run --hook-stage manual --all-files --show-diff-on-failure

set +e

FORMAT_FILE_URL=https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-23.02/cmake-format-rapids-cmake.json
export RAPIDS_CMAKE_FORMAT_FILE=/tmp/rapids_cmake_ci/cmake-formats-rapids-cmake.json
mkdir -p $(dirname ${RAPIDS_CMAKE_FORMAT_FILE})
wget -O ${RAPIDS_CMAKE_FORMAT_FILE} ${FORMAT_FILE_URL}

# Check for a consistent code format
FORMAT=$(python cpp/scripts/run-clang-format.py 2>&1)
FORMAT_RETVAL=$?
if [ "$RETVAL" = "0" ]; then
  RETVAL=$FORMAT_RETVAL
fi

# Output results if failure otherwise show pass
if [ "$FORMAT_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: clang format check; begin output\n\n"
  echo -e "$FORMAT"
  echo -e "\n\n>>>> FAILED: clang format check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: clang format check\n\n"
fi

# clang-tidy check
# NOTE:
#   explicitly pass GPU_ARCHS flag to avoid having to evaluate gpu archs
# because there's no GPU on the CI machine where this script runs!
# NOTE:
#   also, sync all dependencies as they'll be needed by clang-tidy to find
# relevant headers
function setup_and_run_clang_tidy() {
    local LD_LIBRARY_PATH_CACHED=$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
    mkdir cpp/build && \
        cd cpp/build && \
        cmake -DGPU_ARCHS=70 \
              -DBLAS_LIBRARIES=${CONDA_PREFIX}/lib/libopenblas.so.0 \
              .. && \
        make treelite && \
        cd ../.. && \
        python cpp/scripts/run-clang-tidy.py
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH_CACHED
}
TIDY=$(setup_and_run_clang_tidy 2>&1)
TIDY_RETVAL=$?
if [ "$RETVAL" = "0" ]; then
  RETVAL=$TIDY_RETVAL
fi

# Output results if failure otherwise show pass
if [ "$TIDY_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: clang tidy check; begin output\n\n"
  echo -e "$TIDY"
  echo -e "\n\n>>>> FAILED: clang tidy check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: clang tidy check\n\n"
fi


exit $RETVAL

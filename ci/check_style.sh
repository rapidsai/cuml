#!/bin/bash
# Copyright (c) 2020-2022, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Create checks conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
  --output conda \
  --file_key checks \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n checks
conda activate checks

set +e

FORMAT_FILE_URL=https://raw.githubusercontent.com/rapidsai/rapids-cmake/branch-23.02/cmake-format-rapids-cmake.json
export RAPIDS_CMAKE_FORMAT_FILE=/tmp/rapids_cmake_ci/cmake-formats-rapids-cmake.json
mkdir -p $(dirname ${RAPIDS_CMAKE_FORMAT_FILE})
wget -O ${RAPIDS_CMAKE_FORMAT_FILE} ${FORMAT_FILE_URL}

# TODO: enable once pre-commit checks are configured.
# # Run pre-commit checks
# pre-commit run --hook-stage manual --all-files --show-diff-on-failure

# Run flake8 and get results/return code
FLAKE=$(flake8 --config=python/setup.cfg)
RETVAL=$?

# Output results if failure otherwise show pass
if [ "$FLAKE" != "" ]; then
  echo -e "\n\n>>>> FAILED: flake8 style check; begin output\n\n"
  echo -e "$FLAKE"
  echo -e "\n\n>>>> FAILED: flake8 style check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: flake8 style check\n\n"
fi

# Check for copyright headers in the files modified currently
COPYRIGHT=$(python ci/checks/copyright.py --git-modified-only 2>&1)
CR_RETVAL=$?
if [ "$RETVAL" = "0" ]; then
  RETVAL=$CR_RETVAL
fi

# Output results if failure otherwise show pass
if [ "$CR_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: copyright check; begin output\n\n"
  echo -e "$COPYRIGHT"
  echo -e "\n\n>>>> FAILED: copyright check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: copyright check\n\n"
fi

# Check for a consistent #include syntax
# TODO: keep adding more dirs as and when we update the syntax
HASH_INCLUDE=$(python cpp/scripts/include_checker.py \
                     cpp/bench \
                     cpp/comms/mpi/include \
                     cpp/comms/mpi/src \
                     cpp/comms/std/include \
                     cpp/comms/std/src \
                     cpp/include \
                     cpp/examples \
                     cpp/src \
                     cpp/src_prims \
                     cpp/test \
                     2>&1)
HASH_RETVAL=$?
if [ "$RETVAL" = "0" ]; then
  RETVAL=$HASH_RETVAL
fi

# Output results if failure otherwise show pass
if [ "$HASH_RETVAL" != "0" ]; then
  echo -e "\n\n>>>> FAILED: #include check; begin output\n\n"
  echo -e "$HASH_INCLUDE"
  echo -e "\n\n>>>> FAILED: #include check; end output\n\n"
else
  echo -e "\n\n>>>> PASSED: #include check\n\n"
fi

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

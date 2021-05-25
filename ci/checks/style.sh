#!/bin/bash
# Copyright (c) 2019-2021, NVIDIA CORPORATION.
#####################
# cuML Style Tester #
#####################

# Ignore errors and set path
set +e
export PATH=/conda/bin:/usr/local/cuda/bin:$PATH

# Activate common conda env and install any dependencies needed
source activate gdf
cd "$WORKSPACE"
export GIT_DESCRIBE_TAG=`git describe --tags`
export MINOR_VERSION=`echo $GIT_DESCRIBE_TAG | grep -o -E '([0-9]+\.[0-9]+)'`
conda install "ucx-py=${MINOR_VERSION}" "ucx-proc=*=gpu"

# Run flake8 and get results/return code
FLAKE=`flake8 --config=python/setup.cfg`
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
COPYRIGHT=`python ci/checks/copyright.py --git-modified-only 2>&1`
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
HASH_INCLUDE=`python cpp/scripts/include_checker.py \
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
                     2>&1`
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
FORMAT=`python cpp/scripts/run-clang-format.py 2>&1`
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
TIDY=`setup_and_run_clang_tidy 2>&1`
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

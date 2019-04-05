#!/usr/bin/env bash

set -e

if [ "$BUILD_CUML" == '1' ]; then
  echo "Building cuML"
  CUDA_REL=${CUDA:0:3}

  # Only CUDA 10 supports multigpu ols/tsvd, need to separate the conda build command
  if [ "${CUDA:0:2}" == '10' ]; then
    # CUDA 10 release
    CUDA_REL=${CUDA:0:4}
    if [ "$BUILD_ABI" == "1" ]; then
      conda build conda/recipes/cuml-cuda10 --python=${PYTHON}
    else
      conda build conda/recipes/cuml-cuda10 --python=${PYTHON}
    fi

  else
    if [ "$BUILD_ABI" == "1" ]; then
      conda build conda/recipes/cuml --python=${PYTHON}
    else
      conda build conda/recipes/cuml --python=${PYTHON}
    fi

  fi


fi

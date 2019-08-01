#!/usr/bin/env bash

set -e

if [ "$BUILD_CUML" == '1' ]; then
  echo "Building cuML"
  CUDA_REL=${CUDA_VERSION%.*}

  # Only CUDA 10 supports multigpu ols/tsvd, need to separate the conda build command
  if [ "${CUDA_REL%.*}" == '10' ]; then
    # CUDA 10 release
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

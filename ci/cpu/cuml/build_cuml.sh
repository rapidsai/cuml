#!/usr/bin/env bash

set -e

if [ "$BUILD_CUML" == '1' ]; then
  echo "Building cuML"
  CUDA_REL=${CUDA_VERSION%.*}

  if [ "$BUILD_ABI" == "1" ]; then
    conda build conda/recipes/cuml --python=${PYTHON}
  else
    conda build conda/recipes/cuml --python=${PYTHON}
  fi

fi

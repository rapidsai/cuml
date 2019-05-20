#!/usr/bin/env bash

set -e

if [ "$BUILD_LIBCUML" == '1' -o "$BUILD_CUML" == '1' ]; then
  echo "Building libcuml"
  CUDA_REL=${CUDA:0:3}
  if [ "${CUDA:0:2}" == '10' ]; then
    # CUDA 10 release
    CUDA_REL=${CUDA:0:4}
  fi

  conda config --add channels teju85 # for libclang installation
  if [ "$BUILD_ABI" == "1" ]; then
    conda build conda/recipes/libcuml --python=${PYTHON}
  else
    conda build conda/recipes/libcuml --python=${PYTHON}
  fi
fi

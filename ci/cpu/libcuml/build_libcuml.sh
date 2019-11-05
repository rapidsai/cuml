#!/usr/bin/env bash

set -e

if [ "$BUILD_LIBCUML" == '1' -o "$BUILD_CUML" == '1' ]; then
  echo "Building libcuml"
  CUDA_REL=${CUDA_VERSION%.*}

  conda install -n base -c conda-forge conda=4.7 -y
  conda install -c conda-forge conda-build=3.18 -y

  if [ "$BUILD_ABI" == "1" ]; then
    conda build conda/recipes/libcuml --python=${PYTHON}
  else
    conda build conda/recipes/libcuml --python=${PYTHON}
  fi
fi

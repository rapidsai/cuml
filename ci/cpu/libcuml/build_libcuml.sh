#!/usr/bin/env bash

set -e

if [ "$BUILD_LIBCUML" == '1' -o "$BUILD_CUML" == '1' ]; then
  echo "Building libcuml"
  CUDA_REL=${CUDA_VERSION%.*}

  conda clean -i

  if [ "$BUILD_ABI" == "1" ]; then
    conda build conda/recipes/libcuml --python=${PYTHON}
  else
    conda build conda/recipes/libcuml --python=${PYTHON}
  fi
fi

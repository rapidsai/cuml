#!/usr/bin/env bash

set -e

if [ "$BUILD_LIBCUML" == '1' -o "$BUILD_CUML" == '1' ]; then
  echo "Building libcuml"
  conda build conda/recipes/libcuml -c nvidia -c rapidsai/label/branc-0.5-cuda${CUDA} -c numba -c pytorch -c conda-forge -c defaults --python=${PYTHON}
fi
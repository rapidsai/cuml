#!/usr/bin/env bash

set -e

if [ "$BUILD_LIBCUML" == '1' ]; then
  echo "Building libcuml"
  conda build conda-recipes/libcuml -c nvidia -c rapidsai -c numba -c pytorch -c conda-forge -c defaults --python=${PYTHON}
fi
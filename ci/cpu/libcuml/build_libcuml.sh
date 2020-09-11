#!/usr/bin/env bash

set -e

if [ "$BUILD_LIBCUML" == '1' -o "$BUILD_CUML" == '1' ]; then
  echo "Building libcuml"
  CUDA_REL=${CUDA_VERSION%.*}

  conda build  -c rapidsai-nightly/label/testing conda/recipes/libcuml --python=${PYTHON}
fi

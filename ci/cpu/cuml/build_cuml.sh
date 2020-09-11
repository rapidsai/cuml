#!/usr/bin/env bash

set -e

if [ "$BUILD_CUML" == '1' ]; then
  echo "Building cuML"
  CUDA_REL=${CUDA_VERSION%.*}
  conda build -c rapidsai-nightly/label/testing conda/recipes/cuml --python=${PYTHON}

fi

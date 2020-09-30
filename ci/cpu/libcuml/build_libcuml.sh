#!/usr/bin/env bash

set -e

if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
  if [ "$BUILD_LIBCUML" == '1' -o "$BUILD_CUML" == '1' ]; then
    echo "Building libcuml"
    CUDA_REL=${CUDA_VERSION%.*}
    conda build conda/recipes/libcuml --python=${PYTHON}
  fi
else
  if [ "$BUILD_LIBCUML" == '1' ]; then
    conda build conda/recipes/libcuml --dirty --no-remove-work-dir --python=${PYTHON}
  fi
fi
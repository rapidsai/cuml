#!/usr/bin/env bash

set -e

if [ "$BUILD_LIBCUML" == '1' -o "$BUILD_CUML" == '1' ]; then
  echo "Building libcuml"
  CUDA_REL=${CUDA:0:3}
  if [ "${CUDA:0:2}" == '10' ]; then
      # CUDA 10 release
      CUDA_REL=${CUDA:0:4}
  fi
  conda build conda-recipes/libcuml -c nvidia/label/cuda${CUDA_REL} -c rapidsai/label/cuda${CUDA_REL} -c numba -c pytorch -c conda-forge -c defaults --python=${PYTHON}
fi

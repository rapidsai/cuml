#!/usr/bin/env bash

set -e

if [ "$BUILD_CUML" == '1' ]; then
  echo "Building cuML"
  CUDA_REL=${CUDA:0:3}

  # Only CUDA 10 supports multigpu ols/tsvd, need to separate the conda build command
  if [ "${CUDA:0:2}" == '10' ]; then
    # CUDA 10 release
    CUDA_REL=${CUDA:0:4}
    if [ "$BUILD_ABI" == "1" ]; then
      conda build conda/recipes/cuml-cuda10 -c conda-forge -c numba -c rapidsai/label/cuda${CUDA_REL} -c rapidsai-nightly/label/cuda${CUDA_REL} -c nvidia/label/cuda${CUDA_REL} -c pytorch -c defaults --python=${PYTHON}
    else
      conda build conda/recipes/cuml-cuda10 -c conda-forge/label/cf201901 -c numba -c rapidsai/label/cf201901-cuda${CUDA_REL} -c nvidia/label/cf201901-cuda${CUDA_REL} -c pytorch -c defaults --python=${PYTHON}
    fi

  else
    if [ "$BUILD_ABI" == "1" ]; then
      conda build conda/recipes/cuml -c conda-forge -c numba -c rapidsai/label/cuda${CUDA_REL} -c rapidsai-nightly/label/cuda${CUDA_REL} -c nvidia/label/cuda${CUDA_REL} -c pytorch -c defaults --python=${PYTHON}
    else
      conda build conda/recipes/cuml -c conda-forge/label/cf201901 -c numba -c rapidsai/label/cf201901-cuda${CUDA_REL} -c nvidia/label/cf201901-cuda${CUDA_REL} -c pytorch -c defaults --python=${PYTHON}
    fi

  fi


fi

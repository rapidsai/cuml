#!/usr/bin/env bash

set -e

if [ "$BUILD_CUML" == '1' ]; then
    echo "Building cuML"
    CUDA_REL=${CUDA:0:3}
    if [ "${CUDA:0:2}" == '10' ]; then
        # CUDA 10 release
        CUDA_REL=${CUDA:0:4}
    fi
    conda build conda-recipes/cuml -c defaults -c conda-forge -c numba -c rapidsai/label/cuda${CUDA_REL} -c nvidia/label/cuda${CUDA_REL} -c pytorch --python=${PYTHON}
fi

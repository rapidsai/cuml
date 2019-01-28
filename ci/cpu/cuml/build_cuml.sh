#!/usr/bin/env bash

set -e

if [ "$BUILD_CUML" == '1' ]; then
    echo "Building cuML"
    conda build conda/recipes/cuml -c defaults -c conda-forge -c numba -c rapidsai/label/branch-0.5-cuda${CUDA} -c nvidia -c pytorch --python=${PYTHON}
fi
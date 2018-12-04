#!/usr/bin/env bash

set -e

if [ "$BUILD_CUML" == '1' ]; then
    echo "Building cuML"
    conda build conda-recipes/cuml -c defaults -c conda-forge -c numba -c rapidsai -c nvidia -c pytorch --python=${PYTHON}
fi
ß
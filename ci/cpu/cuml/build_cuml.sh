#!/usr/bin/env bash

set -e

if [ "$BUILD_CUML" == '1' ]; then
  echo "Building cuML"
  CUDA_REL=${CUDA_VERSION%.*}
  if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    conda build conda/recipes/cuml --python=${PYTHON}
  else
    conda build -c ci/artifacts/cuml/cpu/conda-bld/ --dirty --no-remove-work-dir conda/recipes/cuml --python=${PYTHON}
  fi

fi

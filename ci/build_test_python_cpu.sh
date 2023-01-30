#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin py-cpu build"

rapids-mamba-retry mambabuild \
  --no-test \
  conda/recipes/cuml-cpu

. /opt/conda/etc/profile.d/conda.sh

rapids-mamba-retry create -n test python=${RAPIDS_PY_VERSION}

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

# todo: use dependenncies file
rapids-mamba-retry install "scikit-learn=1.2" "hdbscan=0.8.29" "umap-learn=0.5.3" "nvtx"

rapids-mamba-retry install --use-local cuml-cpu


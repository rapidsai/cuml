#!/bin/bash
# Copyright (c) 2023-2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-logger "Create clang_tidy conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-dependency-file-generator \
  --output conda \
  --file-key clang_tidy \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n clang_tidy
# Temporarily allow unbound variables for conda activation.
set +u && conda activate clang_tidy && set -u

./build.sh --configure-only libcuml

rapids-logger "Run clang-tidy"

python cpp/scripts/run-clang-tidy.py --config pyproject.toml

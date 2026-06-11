#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

rapids-logger "Create clang_tidy conda environment"
. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

rapids-dependency-file-generator \
  --output conda \
  --file-key clang_tidy \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n clang_tidy
# Temporarily allow unbound variables for conda activation.
set +u && conda activate clang_tidy && set -u

# clang-tidy parses the GCC compile command with clang. Newer conda compilers add
# this GCC-only optimization flag, which clang reports as an error.
for flags_var in CFLAGS CXXFLAGS; do
  if [[ -n "${!flags_var:-}" ]]; then
    export "${flags_var}=$(printf '%s' "${!flags_var}" | sed -E 's/(^|[[:space:]])-fno-merge-constants([[:space:]]|$)/ /g; s/[[:space:]]+/ /g; s/^ //; s/ $//')"
  fi
done

./build.sh --configure-only libcuml

rapids-logger "Run clang-tidy"

python cpp/scripts/run-clang-tidy.py --config pyproject.toml

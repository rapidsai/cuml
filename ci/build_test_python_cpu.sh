#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.

set -euo pipefail

source rapids-env-update

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-logger "Begin py-cpu build"

rapids-mamba-retry mambabuild \
  conda/recipes/cuml-cpu

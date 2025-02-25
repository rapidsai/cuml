#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

# Support invoking run_cuml_singlegpu_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cuml/cuml/tests

python -m pytest -p cudf.pandas -p cuml.experimental.accel --cache-clear "$@"  experimental/accel/

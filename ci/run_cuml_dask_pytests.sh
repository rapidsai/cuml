#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

# Support invoking run_cuml_dask_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cuml/cuml/tests/dask

python -m pytest --cache-clear "$@" .

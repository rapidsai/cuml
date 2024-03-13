#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

# Support invoking run_cuml_dask_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cuml/tests/dask

python -m pytest -s --cache-clear "$@" test_dask_logistic_regression.py

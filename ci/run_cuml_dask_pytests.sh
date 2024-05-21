#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

# Support invoking run_cuml_dask_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cuml/tests/dask

rapids-logger "pytest cuml-dask (No UCX-Py/UCXX)"
timeout 2h python -m pytest --cache-clear "$@" .

rapids-logger "pytest cuml-dask (UCX-Py only)"
timeout 5m python -m pytest --cache-clear --run_ucx "$@" .

rapids-logger "pytest cuml-dask (UCXX only)"
timeout 5m python -m pytest --cache-clear --run_ucxx "$@" .

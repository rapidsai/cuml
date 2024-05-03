#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

# Support invoking run_cuml_dask_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cuml/tests/dask

# Disable non-UCXX tests for wheels tests
if [[ -v CONDA_PREFIX ]]; then
    rapids-logger "pytest cuml-dask (No UCX-Py/UCXX)"
    python -m pytest --cache-clear "$@" .

    rapids-logger "pytest cuml-dask (UCX-Py only)"
    python -m pytest --cache-clear --run_ucx "$@" .
fi

rapids-logger "pytest cuml-dask (UCXX only)"
python -m pytest --cache-clear --run_ucxx "$@" .

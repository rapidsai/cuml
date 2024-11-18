#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

# Support invoking run_cuml_singlegpu_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cuml/cuml/tests

python -m pytest --cache-clear --ignore=dask -m "not memleak" "$@" .

cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cuml/cuml/tests/experimental/accel

python -m pytest -p cuml.experimental.accel --cache-clear "$@" .
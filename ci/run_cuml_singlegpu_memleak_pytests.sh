#!/bin/bash
# Copyright (c) 2024, NVIDIA CORPORATION.

# Support invoking run_cuml_singlegpu_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cuml/tests

pytest --cache-clear --ignore=dask -m "memleak" "$@" .

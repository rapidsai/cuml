#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

source ./ci/use_conda_packages_from_prs.sh


# Support invoking run_cuml_dask_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cuml/cuml/tests/dask || exit 1

python -m pytest --cache-clear "$@" .

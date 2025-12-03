#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

source ./ci/use_conda_packages_from_prs.sh

# Support invoking run_cuml_singlegpu_pytests.sh outside the script directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../python/cuml/tests || exit 1

python -m pytest -p cudf.pandas --cache-clear --ignore=dask "$@" --quick_run .

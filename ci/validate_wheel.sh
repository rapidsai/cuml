#!/bin/bash
# Copyright (c) 2024-2025, NVIDIA CORPORATION.

set -euo pipefail

package_dir=$1
wheel_dir_relative_path=$2

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"

# some packages are much larger on CUDA 11 than on CUDA 12
PYDISTCHECK_ARGS=()
if [[ "${package_dir}" == "python/libcuml" ]]; then
    if [[ "${RAPIDS_CUDA_MAJOR}" == "11" ]]; then
        PYDISTCHECK_ARGS+=(
            --max-allowed-size-compressed '1.0G'
        )
    else
        PYDISTCHECK_ARGS+=(
            --max-allowed-size-compressed '500M'
        )
    fi
fi

cd "${package_dir}"

rapids-logger "validate packages with 'pydistcheck'"

pydistcheck \
    --inspect \
    "${PYDISTCHECK_ARGS[@]}" \
    "$(echo ${wheel_dir_relative_path}/*.whl)"

rapids-logger "validate packages with 'twine'"

twine check \
    --strict \
    "$(echo ${wheel_dir_relative_path}/*.whl)"

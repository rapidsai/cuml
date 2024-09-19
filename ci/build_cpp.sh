#!/bin/bash
# Copyright (c) 2022-2024, NVIDIA CORPORATION.

set -euo pipefail

rapids-configure-conda-channels

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"/../
source ./ci/use_conda_packages_from_prs.sh

rapids-print-env

rapids-logger "Begin cpp build"

RAPIDS_PACKAGE_VERSION=$(rapids-generate-version) rapids-conda-retry mambabuild \
    conda/recipes/libcuml

rapids-upload-conda-to-s3 cpp

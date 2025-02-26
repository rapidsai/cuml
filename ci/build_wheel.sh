#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.

set -euo pipefail

package_name=$1
package_dir=$2

source rapids-configure-sccache
source rapids-date-string
source rapids-telemetry-setup

rapids-generate-version > ./VERSION

cd "${package_dir}"

sccache --zero-stats

rapids-logger "Building '${package_name}' wheel"
rapids-pip-retry wheel \
    -w dist \
    -v \
    --no-deps \
    --disable-pip-version-check \
    . 2>&1 | tee ${GITHUB_WORKSPACE}/telemetry-artifacts/build.log

sccache --show-adv-stats | tee ${GITHUB_WORKSPACE}/telemetry-artifacts/sccache-stats.txt

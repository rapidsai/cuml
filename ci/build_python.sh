#!/bin/bash
# Copyright (c) 2022-2025, NVIDIA CORPORATION.

set -euo pipefail

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

rapids-generate-version > ./VERSION

rapids-logger "Begin py build"

CPP_CHANNEL=$(rapids-download-conda-from-github cpp)

RAPIDS_PACKAGE_VERSION=$(head -1 ./VERSION)
export RAPIDS_PACKAGE_VERSION

# populates `RATTLER_CHANNELS` array and `RATTLER_ARGS` array
source rapids-rattler-channel-string

rapids-logger "Prepending channel ${CPP_CHANNEL} to RATTLER_CHANNELS"

RATTLER_CHANNELS=("--channel" "${CPP_CHANNEL}" "${RATTLER_CHANNELS[@]}")

sccache --zero-stats

rapids-logger "Building cuml"

# --no-build-id allows for caching with `sccache`
# more info is available at
# https://rattler.build/latest/tips_and_tricks/#using-sccache-or-ccache-with-rattler-build
rattler-build build --recipe conda/recipes/cuml \
                    "${RATTLER_ARGS[@]}" \
                    "${RATTLER_CHANNELS[@]}"

sccache --show-adv-stats

# remove build_cache directory to avoid uploading the entire source tree
# tracked in https://github.com/prefix-dev/rattler-build/issues/1424
rm -rf "$RAPIDS_CONDA_BLD_OUTPUT_DIR"/build_cache

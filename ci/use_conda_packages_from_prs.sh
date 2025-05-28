#!/bin/bash
# Copyright (c) 2025, NVIDIA CORPORATION.

set +x

# download CI artifacts
echo "--- getting commit ---"
rapids-retry --quiet gh pr view "256" --repo "rapidsai/cumlprims_mg" --json headRefOid --jq '.headRefOid'

echo "--- downloading artifacts ---"
LIBCUMLPRIMS_MG_CHANNEL=$(rapids-get-pr-conda-artifact cumlprims_mg 256 cpp)
echo "--- done downloading artifacts ---"

# For `rattler` builds:
#
# Add these channels to the array checked by 'rapids-rattler-channel-string'.
# This ensures that when conda packages are built with strict channel priority enabled,
# the locally-downloaded packages will be preferred to remote packages (e.g. nightlies).
#
RAPIDS_PREPENDED_CONDA_CHANNELS=(
    "${LIBCUMLPRIMS_MG_CHANNEL}"
)
export RAPIDS_PREPENDED_CONDA_CHANNELS

# For tests and `conda-build` builds:
#
# Add these channels to the system-wide conda configuration.
# This results in PREPENDING them to conda's channel list, so
# these packages should be found first if strict channel priority is enabled.
#
for _channel in "${RAPIDS_PREPENDED_CONDA_CHANNELS[@]}"
do
   conda config --system --add channels "${_channel}"
done

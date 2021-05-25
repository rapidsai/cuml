#!/bin/bash
# Copyright (c) 2019, NVIDIA CORPORATION.
########################
# cuML Version Updater #
########################

## Usage
# bash update-version.sh <new_version>


# Format is YY.MM.PP - no leading 'v' or trailing 'a'
NEXT_FULL_TAG=$1

# Get current version
CURRENT_TAG=$(git tag | grep -xE 'v[0-9\.]+' | sort --version-sort | tail -n 1 | tr -d 'v')
CURRENT_MAJOR=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[1]}')
CURRENT_MINOR=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[2]}')
CURRENT_PATCH=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[3]}')
CURRENT_SHORT_TAG=${CURRENT_MAJOR}.${CURRENT_MINOR}

#Get <major>.<minor> for next version
NEXT_MAJOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[2]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}

echo "Preparing release $CURRENT_TAG => $NEXT_FULL_TAG"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' $2 && rm -f ${2}.bak
}

sed_runner 's/'"CUML VERSION .* LANGUAGES"'/'"cuML VERSION ${NEXT_FULL_TAG} LANGUAGES"'/g' cpp/CMakeLists.txt
# RTD update
sed_runner 's/version = .*/version = '"'${NEXT_SHORT_TAG}'"'/g' docs/source/conf.py
sed_runner 's/release = .*/release = '"'${NEXT_FULL_TAG}'"'/g' docs/source/conf.py

for FILE in conda/environments/*.yml; do
   sed_runner "s/cudf=${CURRENT_SHORT_TAG}/cudf=${NEXT_SHORT_TAG}/g" ${FILE};
   sed_runner "s/rmm=${CURRENT_SHORT_TAG}/rmm=${NEXT_SHORT_TAG}/g" ${FILE};
   sed_runner "s/dask-cuda=${CURRENT_SHORT_TAG}/dask-cuda=${NEXT_SHORT_TAG}/g" ${FILE};
   sed_runner "s/dask-cudf=${CURRENT_SHORT_TAG}/dask-cudf=${NEXT_SHORT_TAG}/g" ${FILE};
   sed_runner "s/ucx-py=${CURRENT_SHORT_TAG}/ucx-py=${NEXT_SHORT_TAG}/g" ${FILE};
   sed_runner "s/libcumlprims=${CURRENT_SHORT_TAG}/libcumlprims=${NEXT_SHORT_TAG}/g" ${FILE};
   sed_runner "s/rapids-build-env=${CURRENT_SHORT_TAG}/rapids-build-env=${NEXT_SHORT_TAG}/g" ${FILE};
   sed_runner "s/rapids-notebook-env=${CURRENT_SHORT_TAG}/rapids-notebook-env=${NEXT_SHORT_TAG}/g" ${FILE};
   sed_runner "s/rapids-doc-env=${CURRENT_SHORT_TAG}/rapids-doc-env=${NEXT_SHORT_TAG}/g" ${FILE};
done
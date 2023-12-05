#!/bin/bash
# Copyright (c) 2019-2023, NVIDIA CORPORATION.
########################
# cuML Version Updater #
########################

## Usage
# bash update-version.sh <new_version>


# Format is YY.MM.PP - no leading 'v' or trailing 'a'
NEXT_FULL_TAG=$1

# Get current version
CURRENT_TAG=$(git tag --merged HEAD | grep -xE '^v.*' | sort --version-sort | tail -n 1 | tr -d 'v')
CURRENT_MAJOR=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[1]}')
CURRENT_MINOR=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[2]}')
CURRENT_PATCH=$(echo $CURRENT_TAG | awk '{split($0, a, "."); print a[3]}')
CURRENT_SHORT_TAG=${CURRENT_MAJOR}.${CURRENT_MINOR}

#Get <major>.<minor> for next version
NEXT_MAJOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo $NEXT_FULL_TAG | awk '{split($0, a, "."); print a[2]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}
NEXT_UCX_PY_VERSION="$(curl -sL https://version.gpuci.io/rapids/${NEXT_SHORT_TAG}).*"

# Need to distutils-normalize the original version
NEXT_SHORT_TAG_PEP440=$(python -c "from setuptools.extern import packaging; print(packaging.version.Version('${NEXT_SHORT_TAG}'))")

echo "Preparing release $CURRENT_TAG => $NEXT_FULL_TAG"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' $2 && rm -f ${2}.bak
}


# Centralized version file update
echo "${NEXT_FULL_TAG}" > VERSION

# pyproject.toml versions
sed_runner "s/rmm==.*\",/rmm==${NEXT_SHORT_TAG_PEP440}.*\",/g" python/pyproject.toml
sed_runner "s/cudf==.*\",/cudf==${NEXT_SHORT_TAG_PEP440}.*\",/g" python/pyproject.toml
sed_runner "s/pylibraft==.*\",/pylibraft==${NEXT_SHORT_TAG_PEP440}.*\",/g" python/pyproject.toml
sed_runner "s/raft-dask==.*\",/raft-dask==${NEXT_SHORT_TAG_PEP440}.*\",/g" python/pyproject.toml


# CMakeLists
sed_runner 's/'"CUML VERSION .* LANGUAGES"'/'"CUML VERSION ${NEXT_FULL_TAG} LANGUAGES"'/g' cpp/CMakeLists.txt
sed_runner 's/'"set(CUML_VERSION .*)"'/'"set(CUML_VERSION ${NEXT_FULL_TAG})"'/g' python/CMakeLists.txt

# rapids-cmake version
sed_runner 's/'"branch-.*\/RAPIDS.cmake"'/'"branch-${NEXT_SHORT_TAG}\/RAPIDS.cmake"'/g' fetch_rapids.cmake


# RTD update
sed_runner 's/version = .*/version = '"'${NEXT_SHORT_TAG}'"'/g' docs/source/conf.py
sed_runner 's/release = .*/release = '"'${NEXT_FULL_TAG}'"'/g' docs/source/conf.py

# Update project_number (RAPIDS_VERSION) in the CPP doxygen file
sed_runner "s/\(PROJECT_NUMBER.*=\).*/\1 \"${NEXT_SHORT_TAG}\"/g" cpp/Doxyfile.in

# Wheel testing script
sed_runner "s/branch-.*/branch-${NEXT_SHORT_TAG}/g" ci/test_wheel.sh

DEPENDENCIES=(
  cudf
  dask-cuda
  dask-cudf
  libcumlprims
  libraft-headers
  libraft
  librmm
  pylibraft
  raft-dask
  rapids-dask-dependency
  rmm
)
for FILE in dependencies.yaml conda/environments/*.yaml; do
  for DEP in "${DEPENDENCIES[@]}"; do
    sed_runner "/- ${DEP}==/ s/==.*/==${NEXT_SHORT_TAG_PEP440}\.*/g" ${FILE};
  done
done

sed_runner "s|/branch-[^/]*/|/branch-${NEXT_SHORT_TAG}/|g" README.md
sed_runner "s|/branch-[^/]*/|/branch-${NEXT_SHORT_TAG}/|g" python/README.md

# Wheel builds clone cumlprims_mg, update its branch
sed_runner "s/extra-repo-sha: branch-.*/extra-repo-sha: branch-${NEXT_SHORT_TAG}/g" .github/workflows/*.yaml

# CI files
for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-workflows/ s/@.*/@branch-${NEXT_SHORT_TAG}/g" "${FILE}"
done
sed_runner "s/RAPIDS_VERSION_NUMBER=\".*/RAPIDS_VERSION_NUMBER=\"${NEXT_SHORT_TAG}\"/g" ci/build_docs.sh

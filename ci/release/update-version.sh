#!/bin/bash
# Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
sed_runner "s/rmm==.*\",/rmm==${NEXT_SHORT_TAG_PEP440}.*,>=0.0.0a0\",/g" python/pyproject.toml
sed_runner "s/cudf==.*\",/cudf==${NEXT_SHORT_TAG_PEP440}.*,>=0.0.0a0\",/g" python/pyproject.toml
sed_runner "s/pylibraft==.*\",/pylibraft==${NEXT_SHORT_TAG_PEP440}.*,>=0.0.0a0\",/g" python/pyproject.toml
sed_runner "s/raft-dask==.*\",/raft-dask==${NEXT_SHORT_TAG_PEP440}.*,>=0.0.0a0\",/g" python/pyproject.toml

DEPENDENCIES=(
  cudf
  cuml
  dask-cuda
  dask-cudf
  libcuml
  libcuml-tests
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
    sed_runner "/-.* ${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}==/ s/==.*/==${NEXT_SHORT_TAG_PEP440}.*,>=0.0.0a0/g" "${FILE}"
  done
done

sed_runner "s|/branch-[^/]*/|/branch-${NEXT_SHORT_TAG}/|g" README.md
sed_runner "s|/branch-[^/]*/|/branch-${NEXT_SHORT_TAG}/|g" python/README.md


# CI files
for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-workflows/ s/@.*/@branch-${NEXT_SHORT_TAG}/g" "${FILE}"
  # Wheel builds clone cumlprims_mg, update its branch
  sed_runner "s/extra-repo-sha: branch-.*/extra-repo-sha: branch-${NEXT_SHORT_TAG}/g" "${FILE}"
done
sed_runner "s/RAPIDS_VERSION_NUMBER=\".*/RAPIDS_VERSION_NUMBER=\"${NEXT_SHORT_TAG}\"/g" ci/build_docs.sh

# .devcontainer files
find .devcontainer/ -type f -name devcontainer.json -print0 | while IFS= read -r -d '' filename; do
    sed_runner "s@rapidsai/devcontainers:[0-9.]*@rapidsai/devcontainers:${NEXT_SHORT_TAG}@g" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/ucx:[0-9.]*@rapidsai/devcontainers/features/ucx:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/cuda:[0-9.]*@rapidsai/devcontainers/features/cuda:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
    sed_runner "s@rapidsai/devcontainers/features/rapids-build-utils:[0-9.]*@rapidsai/devcontainers/features/rapids-build-utils:${NEXT_SHORT_TAG_PEP440}@" "${filename}"
    sed_runner "s@rapids-\${localWorkspaceFolderBasename}-${CURRENT_SHORT_TAG}@rapids-\${localWorkspaceFolderBasename}-${NEXT_SHORT_TAG}@g" "${filename}"
done

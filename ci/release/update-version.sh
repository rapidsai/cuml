#!/bin/bash
########################
# cuML Version Updater #
########################

## Usage
# bash update-version.sh <type>
#     where <type> is either `major`, `minor`, `patch`

set -ex

# Grab argument for release type
RELEASE_TYPE=$1

# Get current version and calculate next versions
CURRENT_TAG=`git describe --abbrev=0 --tags | tr -d 'v'`
CURRENT_MAJOR=`echo $CURRENT_TAG | awk '{split($0, a, "."); print a[1]}'`
CURRENT_MINOR=`echo $CURRENT_TAG | awk '{split($0, a, "."); print a[2]}'`
CURRENT_PATCH=`echo $CURRENT_TAG | awk '{split($0, a, "."); print a[3]}'`
NEXT_MAJOR=$((CURRENT_MAJOR + 1))
NEXT_MINOR=$((CURRENT_MINOR + 1))
NEXT_PATCH=$((CURRENT_PATCH + 1))
NEXT_TAG=""

# Determine release type
if [ "$RELEASE_TYPE" == "major" ]; then
  NEXT_TAG="${NEXT_MAJOR}.0.0"
elif [ "$RELEASE_TYPE" == "minor" ]; then
  NEXT_TAG="${CURRENT_MAJOR}.${NEXT_MINOR}.0"
elif [ "$RELEASE_TYPE" == "patch" ]; then
  NEXT_TAG="${CURRENT_MAJOR}.${CURRENT_MINOR}.${NEXT_PATCH}"
else
  echo "Incorrect release type; use 'major', 'minor', or 'patch' as an argument"
  exit 1
fi

# Move to root of repo
cd ../..
echo "Preparing '$RELEASE_TYPE' release [$CURRENT_TAG -> $NEXT_TAG]"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' $2 && rm -f ${2}.bak
}

# Conda environment updates
sed_runner 's/cuml=.*/cuml='"${NEXT_TAG}.*"'/g' conda_environments/builddocs_py36.yml

# RTD update
sed_runner 's/version = .*/version = '"'${NEXT_TAG}'"'/g' docs/source/conf.py
sed_runner 's/release = .*/release = '"'${NEXT_TAG}'"'/g' docs/source/conf.py

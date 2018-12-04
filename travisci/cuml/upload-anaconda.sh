#!/bin/bash
#
# Adopted from https://github.com/tmcdonell/travis-scripts/blob/dfaac280ac2082cd6bcaba3217428347899f2975/update-accelerate-buildbot.sh

set -e

if [ "$BUILD_CUML" == "1" ]; then
  export UPLOADFILE=`conda build conda-recipes/cuml -c defaults -c conda-forge -c numba -c rapidsai -c nvidia -c pytorch --python=${PYTHON} --output`
  SOURCE_BRANCH=master

  CUDA_REL=${CUDA:0:3}
  if [ "${CUDA:0:2}" == '10' ]; then
    # CUDA 10 release
    CUDA_REL=${CUDA:0:4}
  fi

  LABEL_OPTION="--label dev --label cuda${CUDA_REL}"
  if [ "${LABEL_MAIN}" == '1' ]; then
    LABEL_OPTION="--label main --label cuda${CUDA_REL}"
  fi
  LABEL_OPTION="--label test --label cuda${CUDA_REL}"
  echo "LABEL_OPTION=${LABEL_OPTION}"

  test -e ${UPLOADFILE}

  # Pull requests or commits to other branches shouldn't upload
  if [ ${TRAVIS_PULL_REQUEST} != false -o ${TRAVIS_BRANCH} != ${SOURCE_BRANCH} ]; then
    echo "Skipping upload"
    return 0
  fi

  if [ -z "$MY_UPLOAD_KEY" ]; then
    echo "No upload key"
    return 0
  fi

  echo "Upload"
  echo ${UPLOADFILE}
  anaconda -t ${MY_UPLOAD_KEY} upload -u rapidsai ${LABEL_OPTION} --force ${UPLOADFILE}
fi

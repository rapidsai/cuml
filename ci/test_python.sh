#!/bin/bash

set -euo pipefail

#TODO: Remove
. /opt/conda/etc/profile.d/conda.sh
conda activate base

# Check environment
source ci/check_env.sh

# GPU Test Stage
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

gpuci_mamba_retry install \
  -c "${CPP_CHANNEL}" \
  -c "${PYTHON_CHANNEL}" \
  libcuml cuml

# Install test dependencies
gpuci_mamba_retry install pytest pytest-cov
gpuci_logger "Install the main version of dask, distributed, and dask-glm"
pip install "git+https://github.com/dask/distributed.git@2022.7.1" --upgrade --no-deps
pip install "git+https://github.com/dask/dask.git@2022.7.1" --upgrade --no-deps
pip install "git+https://github.com/dask/dask-glm@main" --force-reinstall --no-deps
pip install sparse

gpuci_logger "Check GPU usage"
nvidia-smi

gpuci_logger "Python pytest for cuml"
cd "${GITHUB_WORKSPACE}/python/cuml/tests"

pytest --cache-clear --basetemp="${GITHUB_WORKSPACE}/cuml-cuda-tmp" --junitxml="${GITHUB_WORKSPACE}/junit-cuml.xml" -v -s -m "not memleak" --durations=50 --timeout=300 --ignore=dask --cov-config=.coveragerc --cov=cuml --cov-report="xml:${GITHUB_WORKSPACE}/python/cuml/cuml-coverage.xml" --cov-report term

timeout 7200 sh -c "pytest dask --cache-clear --basetemp=${GITHUB_WORKSPACE}/cuml-mg-cuda-tmp --junitxml=${GITHUB_WORKSPACE}/junit-cuml-mg.xml -v -s -m 'not memleak' --durations=50 --timeout=300 --cov-config=.coveragerc --cov=cuml --cov-report=xml:${GITHUB_WORKSPACE}/python/cuml/cuml-dask-coverage.xml --cov-report term"

gpuci_logger "Notebook tests"
set +e
EXITCODE=0
trap "EXITCODE=1" ERR
"${GITHUB_WORKSPACE}/ci/test_notebooks.sh" 2>&1 | tee nbtest.log
python "${GITHUB_WORKSPACE}/ci/utils/nbtestlog2junitxml.py nbtest.log"

CODECOV_TOKEN="${CODECOV_TOKEN:-}"
if [ -n "${CODECOV_TOKEN}" ]; then
  # NOTE: The code coverage upload needs to work for both PR builds and normal
  # branch builds (aka `branch-0.XX`). Ensure the following settings to the
  # codecov CLI will work with and without a PR
  gpuci_logger "Uploading Code Coverage to codecov.io"

  # Directory containing reports
  REPORT_DIR="${GITHUB_WORKSPACE}/python/cuml"

  # Base name to use in Codecov UI
  CODECOV_NAME=${OS},py${PY_VER},cuda${CUDA_VER}

  # Codecov args needed by both calls
  EXTRA_CODECOV_ARGS="-c"

  # Save the OS PYTHON and CUDA flags
  EXTRA_CODECOV_ARGS="${EXTRA_CODECOV_ARGS} -e OS,PY_VER,CUDA_VER"

  # If we have GIT_SHA, use that instead. This fixes an issue where
  # CodeCov uses a local merge commit created by GitHub. Since this commit
  # never gets pushed, it causes issues in Codecov
  if [ -n "${GIT_SHA}" ]; then
      EXTRA_CODECOV_ARGS="${EXTRA_CODECOV_ARGS} -C ${GIT_SHA}"
  fi

  # Get PR ID from GITHUB_REF. GITHUB_REF looks like `refs/pull/PR_ID/merge`
  if [ -n "${GITHUB_REF}" ]; then
    PR_ID="$(echo ${GITHUB_REF} | cut -d '/' -f 3)"
    EXTRA_CODECOV_ARGS="${EXTRA_CODECOV_ARGS} -P ${PR_ID}"
  fi

  # Set the slug since this does not work in jenkins.
  export CODECOV_SLUG="${GITHUB_ACTOR:-"rapidsai"}/cuml"

  # Upload the two reports with separate flags. Delete the report on success
  # to prevent further CI steps from re-uploading
  curl -s https://codecov.io/bash | bash -s -- -F non-dask -f ${REPORT_DIR}/cuml-coverage.xml -n "${CODECOV_NAME},non-dask" ${EXTRA_CODECOV_ARGS}
  curl -s https://codecov.io/bash | bash -s -- -F dask -f ${REPORT_DIR}/cuml-dask-coverage.xml -n "${CODECOV_NAME},dask" ${EXTRA_CODECOV_ARGS}
fi

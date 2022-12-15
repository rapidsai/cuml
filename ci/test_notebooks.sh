#!/bin/bash
# Copyright (c) 2020-2022, NVIDIA CORPORATION.
set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Generate Python testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file_key test_notebooks \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --force -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
RAPIDS_COVERAGE_DIR=${RAPIDS_COVERAGE_DIR:-"${PWD}/coverage-results"}
mkdir -p "${RAPIDS_TESTS_DIR}" "${RAPIDS_COVERAGE_DIR}"
SUITEERROR=0

rapids-print-env

rapids-mamba-retry install \
  -c "${CPP_CHANNEL}" \
  -c "${PYTHON_CHANNEL}" \
  cuml

rapids-logger "Check GPU usage"
nvidia-smi

set +e

rapids-logger "notebook tests cuml"

# TODO: The WORKSPACE var can probably be removed.
WORKSPACE=${WORKSPACE:-$PWD}
NOTEBOOKS_DIR="$WORKSPACE/notebooks"
NOTEBOOKS_DIR="notebooks"
NBTEST="$WORKSPACE/ci/utils/nbtest.sh"
LIBCUDF_KERNEL_CACHE_PATH="$WORKSPACE/.jitcache"

pushd "${NOTEBOOKS_DIR}"
TOPLEVEL_NB_FOLDERS=$(find . -name "*.ipynb" |cut -d'/' -f2|sort -u)

# Add notebooks that should be skipped here
# (space-separated list of filenames without paths)

SKIPNBS="cuml_benchmarks.ipynb"

NOTEBOOKS_EXITCODE=0

# Always run nbtest in all TOPLEVEL_NB_FOLDERS, set EXITCODE to failure
# if any run fails

for nb in $(find . -name "*.ipynb"); do
    nbBasename=$(basename ${nb})
    # Skip all NBs that use dask (in the code or even in their name)
    if ((echo ${nb}|grep -qi dask) || \
        (grep -q dask ${nb})); then
        echo "--------------------------------------------------------------------------------"
        echo "SKIPPING: ${nb} (suspected Dask usage, not currently automatable)"
        echo "--------------------------------------------------------------------------------"
    elif (echo " ${SKIPNBS} " | grep -q " ${nbBasename} "); then
        echo "--------------------------------------------------------------------------------"
        echo "SKIPPING: ${nb} (listed in skip list)"
        echo "--------------------------------------------------------------------------------"
    else
        nvidia-smi
        ${NBTEST} ${nbBasename}
        NOTEBOOKS_EXITCODE=$((NOTEBOOKS_EXITCODE | $?))
        rm -rf ${LIBCUDF_KERNEL_CACHE_PATH}/*
    fi
done
popd


nvidia-smi

exit ${NOTEBOOKS_EXITCODE}

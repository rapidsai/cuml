#!/bin/bash
# Copyright (c) 2020-2024, NVIDIA CORPORATION.
set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

LIBRMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1678 cpp)
RMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1678 python)

UCXX_CHANNEL=$(rapids-get-pr-conda-artifact ucxx 278 cpp)

LIBCUDF_CHANNEL=$(rapids-get-pr-conda-artifact cudf 16806 cpp)
CUDF_CHANNEL=$(rapids-get-pr-conda-artifact cudf 16806 python)

LIBRAFT_CHANNEL=$(rapids-get-pr-conda-artifact raft 2433 cpp)
RAFT_CHANNEL=$(rapids-get-pr-conda-artifact raft 2433 python)

rapids-logger "Generate Notebook testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_notebooks \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
  --prepend-channel "${CPP_CHANNEL}" \
  --prepend-channel "${PYTHON_CHANNEL}" \
  --prepend-channel "${LIBRMM_CHANNEL}" \
  --prepend-channel "${RMM_CHANNEL}" \
  --prepend-channel "${UCXX_CHANNEL}" \
  --prepend-channel "${LIBCUDF_CHANNEL}" \
  --prepend-channel "${CUDF_CHANNEL}" \
  --prepend-channel "${LIBRAFT_CHANNEL}" \
  --prepend-channel "${RAFT_CHANNEL}" \
  --prepend-channel "${CUMLPRIMS_CHANNEL}" \
| tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-print-env

rapids-logger "Check GPU usage"
nvidia-smi

NOTEBOOKS_EXITCODE=0
trap "NOTEBOOKS_EXITCODE=1" ERR
set +e

rapids-logger "notebook tests cuml"

# Add notebooks that should be skipped here
# (space-separated list of filenames without paths)
SKIPNBS="cuml_benchmarks.ipynb"
NBTEST="$(realpath "$(dirname "$0")/utils/nbtest.sh")"

cd notebooks
for nb in $(find . -name "*.ipynb"); do
    nbBasename=$(basename "${nb}")
    # Skip all NBs that use dask (in the code or even in their name)
    if ((echo "${nb}" | grep -qi dask) || \
        (grep -q dask "${nb}")); then
        echo "--------------------------------------------------------------------------------"
        echo "SKIPPING: ${nb} (suspected Dask usage, not currently automatable)"
        echo "--------------------------------------------------------------------------------"
    elif (echo " ${SKIPNBS} " | grep -q " ${nbBasename} "); then
        echo "--------------------------------------------------------------------------------"
        echo "SKIPPING: "${nb}" (listed in skip list)"
        echo "--------------------------------------------------------------------------------"
    else
        nvidia-smi
        ${NBTEST} "${nbBasename}"
    fi
done

rapids-logger "Test script exiting with value: $NOTEBOOKS_EXITCODE"
exit ${NOTEBOOKS_EXITCODE}

#!/bin/bash

LIBRMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1678 cpp)
RMM_CHANNEL=$(rapids-get-pr-conda-artifact rmm 1678 python)

CUDF_CPP_CHANNEL=$(rapids-get-pr-conda-artifact cudf 16806 cpp)
CUDF_PYTHON_CHANNEL=$(rapids-get-pr-conda-artifact cudf 16806 python)

UCXX_CHANNEL=$(rapids-get-pr-conda-artifact ucxx 278 cpp)

LIBRAFT_CHANNEL=$(rapids-get-pr-conda-artifact raft 2433 cpp)
RAFT_CHANNEL=$(rapids-get-pr-conda-artifact raft 2433 python)

# NOTE: cloning private repos with rapids-get-pr-conda-artifact doesn't work
# CUMLPRIMS_CHANNEL=$(rapids-get-pr-conda-artifact cumlprims_mg 211 cpp)

conda config --system --add channels "${LIBRMM_CHANNEL}"
conda config --system --add channels "${RMM_CHANNEL}"
conda config --system --add channels "${CUDF_CPP_CHANNEL}"
conda config --system --add channels "${CUDF_PYTHON_CHANNEL}"
conda config --system --add channels "${UCXX_CHANNEL}"
conda config --system --add channels "${LIBRAFT_CHANNEL}"
conda config --system --add channels "${RAFT_CHANNEL}"
# conda config --system --add channels "${CUMLPRIMS_CHANNEL}"

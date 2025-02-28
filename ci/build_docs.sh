#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.
set -euo pipefail

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-s3 cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-s3 python)

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

export RAPIDS_VERSION_MAJOR_MINOR="$(rapids-version-major-minor)"

rapids-dependency-file-generator \
  --output conda \
  --file-key docs \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
  --prepend-channel "${CPP_CHANNEL}" \
  --prepend-channel "${PYTHON_CHANNEL}" \
  | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n docs
conda activate docs

rapids-print-env

export RAPIDS_DOCS_DIR="$(mktemp -d)"

rapids-logger "Build CPP docs"
pushd cpp
doxygen Doxyfile.in
mkdir -p "${RAPIDS_DOCS_DIR}/libcuml/html"
mv html/* "${RAPIDS_DOCS_DIR}/libcuml/html"
popd

rapids-logger "Build Python docs"
pushd docs
sphinx-build -b dirhtml ./source _html -W
mkdir -p "${RAPIDS_DOCS_DIR}/cuml/html"
mv _html/* "${RAPIDS_DOCS_DIR}/cuml/html"
popd

RAPIDS_VERSION_NUMBER="${RAPIDS_VERSION_MAJOR_MINOR}" rapids-upload-docs

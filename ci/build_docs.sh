#!/bin/bash
# Copyright (c) 2023-2025, NVIDIA CORPORATION.
source ./ci/use_conda_packages_from_prs.sh

set -euo pipefail

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-github cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-github python)

rapids-logger "Compressing repodata.json → repodata.json.zst for local channels"
for ch in "${CUVS_CHANNEL:-}" "${LIBCUVS_CHANNEL:-}" "${CPP_CHANNEL:-}"; do
  if [ -n "$ch" ] && [ -d "$ch" ]; then
    for subdir in "$ch"/linux-aarch64 "$ch"/linux-64 "$ch"/noarch; do
      if [ -f "${subdir}/repodata.json" ] && [ ! -f "${subdir}/repodata.json.zst" ]; then
        echo "Compressing ${subdir}/repodata.json → ${subdir}/repodata.json.zst"
        zstd -q -19 "${subdir}/repodata.json" -o "${subdir}/repodata.json.zst"
      fi
    done
  fi
done

echo "=== DEBUG: Listing contents of downloaded conda channels ==="
for ch in "${CUVS_CHANNEL:-}" "${LIBCUVS_CHANNEL:-}" "${CPP_CHANNEL:-}"; do
  if [ -n "$ch" ] && [ -d "$ch" ]; then
    echo ">>> $ch"
    find "$ch" -maxdepth 2 -type f | sort
  else
    echo "!!! Channel path $ch does not exist or is empty"
  fi
done
echo "============================================================="

rapids-logger "Create test conda environment"
. /opt/conda/etc/profile.d/conda.sh

RAPIDS_VERSION_MAJOR_MINOR="$(rapids-version-major-minor)"
export RAPIDS_VERSION_MAJOR_MINOR

rapids-dependency-file-generator \
  --output conda \
  --file-key docs \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" \
  --prepend-channel "${LIBCUVS_CHANNEL}" \
  --prepend-channel "${CUVS_CHANNEL}" \
  --prepend-channel "${CPP_CHANNEL}" \
  --prepend-channel "${PYTHON_CHANNEL}" \
  | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n docs
conda activate docs

rapids-print-env

RAPIDS_DOCS_DIR="$(mktemp -d)"
export RAPIDS_DOCS_DIR

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

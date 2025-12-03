#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

source ./ci/use_wheels_from_prs.sh

RAPIDS_PY_CUDA_SUFFIX="$(rapids-wheel-ctk-name-gen "${RAPIDS_CUDA_VERSION}")"
CUML_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="cuml_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github python)
LIBCUML_WHEELHOUSE=$(RAPIDS_PY_WHEEL_NAME="libcuml_${RAPIDS_PY_CUDA_SUFFIX}" rapids-download-wheels-from-github cpp)
RAPIDS_TESTS_DIR=${RAPIDS_TESTS_DIR:-"${PWD}/test-results"}
mkdir -p "${RAPIDS_TESTS_DIR}"

EXITCODE=0
trap "EXITCODE=1" ERR
set +e

#
# BERTopic Integration Test
#
rapids-logger "===== Testing BERTopic Integration ====="

# Step 1: Install cuML wheels first (two-step workaround for issue #7374)
rapids-logger "Installing cuML wheels"
rapids-pip-retry install \
  "${LIBCUML_WHEELHOUSE}"/libcuml*.whl \
  "${CUML_WHEELHOUSE}"/cuml*.whl

# Step 2: Install BERTopic
rapids-logger "Installing BERTopic"
rapids-pip-retry install bertopic

# Test 1: Verify imports
rapids-logger "Testing imports"
python -c "
import cuml
import bertopic
print('✓ Import test passed')
"

# Test 2: Run minimal end-to-end example
rapids-logger "Running BERTopic end-to-end smoke test"
timeout 20m python -c "
import warnings
warnings.filterwarnings('ignore')

from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups

# Create a small sample dataset
docs = fetch_20newsgroups(subset='train', categories=['sci.space'])['data'][:100]

# Initialize BERTopic with cuML UMAP backend
# BERTopic will automatically use cuML's UMAP if available
topic_model = BERTopic(verbose=False, calculate_probabilities=False)

# Fit the model
topics, probs = topic_model.fit_transform(docs)

print(f'✓ BERTopic smoke test passed - processed {len(docs)} documents, found {len(set(topics))} topics')
"

rapids-logger "===== BERTopic Integration Test Complete ====="

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

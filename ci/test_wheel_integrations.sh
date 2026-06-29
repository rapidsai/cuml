#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

source rapids-init-pip

LIBCUML_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcuml cuml --cuda "$RAPIDS_CUDA_VERSION")")
CUML_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_python cuml cuml --stable --cuda "$RAPIDS_CUDA_VERSION")")
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
  --prefer-binary \
  "${LIBCUML_WHEELHOUSE}"/libcuml*.whl \
  "${CUML_WHEELHOUSE}"/cuml*.whl

# Step 2: Install BERTopic
rapids-logger "Installing BERTopic"
rapids-pip-retry install --prefer-binary bertopic

# Test 1: Verify imports
rapids-logger "Testing imports"
python -c "
import cuml
import bertopic
print('✓ Import test passed')
"

# Test 2: Run minimal end-to-end example
rapids-logger "Running BERTopic end-to-end smoke test (cuml.accel)"
timeout -v 20m python -m cuml.accel -c "
import warnings
warnings.filterwarnings('ignore')

import random
from bertopic import BERTopic

import cuml.accel

# Generate synthetic documents with topic-like word clusters
random.seed(42)
topics = [
    ['star', 'galaxy', 'planet', 'orbit', 'telescope', 'nasa', 'astronaut'],
    ['rocket', 'launch', 'satellite', 'mission', 'space', 'shuttle', 'station'],
    ['moon', 'mars', 'jupiter', 'asteroid', 'comet', 'meteor', 'crater'],
]

docs = []
for i in range(100):
    topic_words = topics[i % len(topics)]
    doc = ' '.join(random.choices(topic_words, k=random.randint(10, 30)))
    docs.append(doc)

topic_model = BERTopic(verbose=False, calculate_probabilities=False)

# Fit the model
topics, probs = topic_model.fit_transform(docs)

# Verify cuML actually backed BERTopic's UMAP and HDBSCAN steps.
assert cuml.accel.enabled(), 'cuml.accel is not enabled'
assert cuml.accel.is_proxy(topic_model.umap_model), \
    f'UMAP not accelerated by cuML: {type(topic_model.umap_model)}'
assert cuml.accel.is_proxy(topic_model.hdbscan_model), \
    f'HDBSCAN not accelerated by cuML: {type(topic_model.hdbscan_model)}'

print(f'✓ BERTopic smoke test passed - processed {len(docs)} documents, found {len(set(topics))} topics')
"

rapids-logger "===== BERTopic Integration Test Complete ====="

rapids-logger "Test script exiting with value: $EXITCODE"
exit ${EXITCODE}

# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import dask
import numpy as np
from scipy import sparse

from cuml.dask.common import to_sparse_dask_array


def load_text_corpus(client):
    """Generate a sparse text-like dataset similar to 20 newsgroups.

    This function generates a sparse bag-of-words matrix and target vector
    that mimic the characteristics of the 20 newsgroups dataset (4 categories)
    after HashingVectorizer transformation.
    """
    n_docs = 2257  # Similar to 20 newsgroups with 4 categories
    n_features = 10000
    n_classes = 4
    avg_nonzero_per_doc = 150

    rng = np.random.RandomState(42)

    # Class labels (balanced)
    y = rng.randint(0, n_classes, size=n_docs)

    # Class-specific word distributions (topic-like)
    class_word_probs = []
    for _ in range(n_classes):
        alpha = np.ones(n_features) * 0.01
        topic = rng.dirichlet(alpha)
        class_word_probs.append(topic)
    class_word_probs = np.vstack(class_word_probs)

    # Generate sparse bag-of-words for each document
    data = []
    rows = []
    cols = []

    for i in range(n_docs):
        label = y[i]
        doc_len = max(1, rng.poisson(avg_nonzero_per_doc))
        word_indices = rng.choice(
            n_features,
            size=doc_len,
            replace=True,
            p=class_word_probs[label],
        )
        unique, counts = np.unique(word_indices, return_counts=True)
        rows.extend([i] * len(unique))
        cols.extend(unique.tolist())
        data.extend(counts.tolist())

    xformed = sparse.csr_matrix(
        (data, (rows, cols)),
        shape=(n_docs, n_features),
        dtype=np.float32,
    )

    X = to_sparse_dask_array(xformed, client)

    y = dask.array.from_array(y, asarray=False, fancy=False).astype(cp.int32)

    return X, y

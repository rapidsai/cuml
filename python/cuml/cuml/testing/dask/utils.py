# SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import dask
import numpy as np

from cuml.dask.common import to_sparse_dask_array
from cuml.testing.datasets import make_text_classification_dataset


def load_text_corpus(client):
    """Generate a sparse text-like dataset similar to 20 newsgroups.

    This function generates a sparse bag-of-words matrix and target vector
    that mimic the characteristics of the 20 newsgroups dataset (4 categories)
    as a distributed dask array.

    Parameters
    ----------
    client : distributed.Client
        Dask distributed client

    Returns
    -------
    tuple
        (X, y) where X is a sparse dask array and y is a dask array
    """
    X, y = make_text_classification_dataset(
        n_docs=2257,  # Similar to 20 newsgroups with 4 categories
        n_classes=4,
        apply_tfidf=False,
        dtype=np.float32,
        random_state=42,
    )

    X = to_sparse_dask_array(X, client)
    y = dask.array.from_array(y, asarray=False, fancy=False).astype(cp.int32)

    return X, y

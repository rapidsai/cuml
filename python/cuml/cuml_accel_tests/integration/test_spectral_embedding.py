#
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

from sklearn.datasets import make_blobs
from sklearn.manifold import SpectralEmbedding


def test_spectral_embedding_make_blobs():
    """Test SpectralEmbedding with make_blobs dataset."""
    X, _ = make_blobs(n_samples=100, centers=3, n_features=20, random_state=42)
    model = SpectralEmbedding(n_components=2, random_state=42)
    X_embedded = model.fit_transform(X)

    assert X_embedded.shape == (100, 2)

# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cudf
import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.manifold import trustworthiness as sklearn_trustworthiness
from umap import UMAP

from cuml.metrics import trustworthiness as cuml_trustworthiness


@pytest.mark.parametrize("input_type", ["ndarray", "dataframe"])
@pytest.mark.parametrize("n_samples", [150, 500])
@pytest.mark.parametrize("n_features", [10, 100])
@pytest.mark.parametrize("n_components", [2, 8])
@pytest.mark.parametrize("batch_size", [128, 1024])
@pytest.mark.filterwarnings(
    "ignore:n_jobs value.*overridden.*by setting random_state.*:UserWarning"
)
# Ignore FutureWarning from third-party umap-learn package calling
# sklearn.utils.validation.check_array with deprecated 'force_all_finite'
# parameter. Old versions of umap-learn use a deprecated parameter.
@pytest.mark.filterwarnings(
    "ignore:'force_all_finite' was renamed to "
    "'ensure_all_finite':FutureWarning:sklearn"
)
def test_trustworthiness(
    input_type, n_samples, n_features, n_components, batch_size
):
    centers = round(n_samples * 0.4)
    X, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        n_features=n_features,
        random_state=32,
    )

    X_embedded = UMAP(
        n_components=n_components, random_state=32
    ).fit_transform(X)
    X = X.astype(np.float32)
    X_embedded = X_embedded.astype(np.float32)

    sk_score = sklearn_trustworthiness(X, X_embedded)

    if input_type == "dataframe":
        X = cudf.DataFrame(X)
        X_embedded = cudf.DataFrame(X_embedded)

    cu_score = cuml_trustworthiness(X, X_embedded, batch_size=batch_size)

    assert abs(cu_score - sk_score) <= 1e-3


def test_trustworthiness_invalid_input():
    X, y = make_blobs(n_samples=10, centers=1, n_features=2, random_state=32)

    with pytest.raises(ValueError):
        cuml_trustworthiness(X, X, n_neighbors=50)

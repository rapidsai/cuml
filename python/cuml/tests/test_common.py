#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np
import pytest

import cuml
from cuml.datasets import make_blobs


@pytest.mark.parametrize(
    "Estimator",
    [
        cuml.KMeans,
        cuml.RandomForestRegressor,
        cuml.RandomForestClassifier,
        cuml.TSNE,
        cuml.UMAP,
    ],
)
@pytest.mark.filterwarnings("ignore:The number of bins.*:UserWarning")
def test_random_state_argument(Estimator):
    X, y = make_blobs(random_state=0)
    # Check that both integer and np.random.RandomState are accepted
    for seed in (42, np.random.RandomState(42)):
        est = Estimator(random_state=seed)

        if est.__class__.__name__ != "TSNE":
            est.fit(X, y)
        else:
            est.fit(X)

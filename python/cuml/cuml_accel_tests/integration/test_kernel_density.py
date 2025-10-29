#
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.neighbors import KernelDensity


@pytest.mark.parametrize("bandwidth", [1.0, 5.0, "scott", "silverman"])
@pytest.mark.parametrize("kernel", ["gaussian", "tophat"])
@pytest.mark.parametrize("metric", ["l1", "l2"])
def test_kernel_density(bandwidth, kernel, metric):
    # Correctness is tested elsewhere, here we're just checking plumbing
    X, _ = make_blobs(random_state=42)
    model = KernelDensity(bandwidth=bandwidth, kernel=kernel, metric=metric)
    model.fit(X)

    scores = model.score_samples(X)
    assert scores.shape == (X.shape[0],)
    assert scores.dtype == np.float64

    score = model.score(X)
    assert isinstance(score, float)

    samples = model.sample(10, random_state=42)
    assert samples.shape == (10, X.shape[1])
    assert samples.dtype == np.float64

    assert isinstance(model.bandwidth_, float)

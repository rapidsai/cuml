# Copyright (c) 2019-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import cupy as cp
import pytest

import cuml

# Testing parameters for scalar parameter tests

dtype = ["single", "double"]

n_samples = [100, 1000]

n_features = [2, 10, 100]

centers = [
    None,
    2,
    5,
]

cluster_std = [0.01, 0.1]

center_box = [
    (-10.0, 10.0),
    [-20.0, 20.0],
]

shuffle = [True, False]


random_state = [None, 9]


@pytest.mark.parametrize("dtype", dtype)
@pytest.mark.parametrize("n_samples", n_samples)
@pytest.mark.parametrize("n_features", n_features)
@pytest.mark.parametrize("centers", centers)
@pytest.mark.parametrize("cluster_std", cluster_std)
@pytest.mark.parametrize("center_box", center_box)
@pytest.mark.parametrize("shuffle", shuffle)
@pytest.mark.parametrize("random_state", random_state)
@pytest.mark.parametrize("order", ["F", "C"])
def test_make_blobs_scalar_parameters(
    dtype,
    n_samples,
    n_features,
    centers,
    cluster_std,
    center_box,
    shuffle,
    random_state,
    order,
):

    out, labels = cuml.make_blobs(
        dtype=dtype,
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=0.001,
        center_box=center_box,
        shuffle=shuffle,
        random_state=random_state,
        order=order,
    )

    assert out.shape == (n_samples, n_features), "out shape mismatch"
    assert labels.shape == (n_samples,), "labels shape mismatch"

    if order == "F":
        assert out.flags["F_CONTIGUOUS"]
    elif order == "C":
        assert out.flags["C_CONTIGUOUS"]

    if centers is None:
        assert cp.unique(labels).shape == (3,), "unexpected number of clusters"
    elif centers <= n_samples:
        assert cp.unique(labels).shape == (
            centers,
        ), "unexpected number of clusters"

# Copyright (c) 2019, NVIDIA CORPORATION.
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

import cuml
import pytest
import numpy as np

from sklearn.metrics import adjusted_rand_score


# Testing parameters for scalar parameter tests

dtype = [
    'single',
    'double'
]

n_samples = [100, 1000]

n_features = [
    2,
    10,
    100
]

centers = [
    None,
    2,
    50,
]

cluster_std = [
    0.01,
    0.1
]

center_box = [
    (-10.0, 10.0),
    [-20.0, 20.0],
]

shuffle = [
    True,
    False
]


random_state = [
    None,
    9
]


@pytest.mark.parametrize('dtype', dtype)
@pytest.mark.parametrize('n_samples', n_samples)
@pytest.mark.parametrize('n_features', n_features)
@pytest.mark.parametrize('centers', centers)
@pytest.mark.parametrize('cluster_std', cluster_std)
@pytest.mark.parametrize('center_box', center_box)
@pytest.mark.parametrize('shuffle', shuffle)
@pytest.mark.parametrize('random_state', random_state)
def test_make_blobs_scalar_parameters(dtype, n_samples, n_features, centers,
                                      cluster_std, center_box, shuffle,
                                      random_state):

    out, labels = cuml.make_blobs(dtype=dtype, n_samples=n_samples,
                                  n_features=n_features, centers=centers,
                                  cluster_std=0.001,
                                  center_box=center_box, shuffle=shuffle,
                                  random_state=random_state)

    # we can use cupy in the future
    labels_np = labels.copy_to_host()

    assert out.shape == (n_samples, n_features), "out shape mismatch"
    assert labels.shape == (n_samples,), "labels shape mismatch"

    if centers is None:
        assert np.unique(labels_np).shape == (3,), \
            "unexpected number of clusters"
    elif centers <= n_samples:
        assert np.unique(labels_np).shape == (centers,), \
            "unexpected number of clusters"


# Parameters for array tests
n_features_ary = [
    2,
    100
]

centers_ary = [
    np.random.uniform(size=(10, 2)),
    np.random.uniform(size=(10, 100))
]


@pytest.mark.parametrize('dtype', dtype)
@pytest.mark.parametrize('n_samples', n_samples)
@pytest.mark.parametrize('n_features', n_features_ary)
@pytest.mark.parametrize('centers', centers_ary)
@pytest.mark.parametrize('cluster_std', cluster_std)
@pytest.mark.parametrize('center_box', center_box)
@pytest.mark.parametrize('shuffle', shuffle)
@pytest.mark.parametrize('random_state', random_state)
def test_make_blobs_ary_parameters(dtype, n_samples, n_features,
                                   centers, cluster_std, center_box,
                                   shuffle, random_state):

    centers = centers.astype(np.dtype(dtype))
    cluster_std = np.full(shape=(1, 10), fill_value=cluster_std, dtype=dtype)

    if centers.shape[1] != n_features or \
            cluster_std.shape[1] != centers.shape[0]:
        with pytest.raises(ValueError):
            out, labels = \
                cuml.make_blobs(dtype=dtype, n_samples=n_samples,
                                n_features=n_features, centers=centers,
                                cluster_std=cluster_std,
                                center_box=center_box, shuffle=shuffle,
                                random_state=random_state)

    else:

        out, labels = \
            cuml.make_blobs(dtype=dtype, n_samples=n_samples,
                            n_features=n_features, centers=centers,
                            cluster_std=cluster_std,
                            center_box=center_box, shuffle=shuffle,
                            random_state=random_state)

        assert out.shape == (n_samples, n_features), "out shape mismatch"
        assert labels.shape == (n_samples,), "labels shape mismatch"

        labels_np = labels.copy_to_host()
        out_np = out.copy_to_host()

        assert np.unique(labels_np).shape == (centers.shape[0],), \
            "unexpected number of clusters"

        # Use kmeans to verify k cluster centers
        from sklearn.cluster import KMeans
        model = KMeans(n_clusters=centers.shape[0])
        model.fit(np.array(out_np))

        assert adjusted_rand_score(model.labels_, labels_np)

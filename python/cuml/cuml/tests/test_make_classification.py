#
# Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

from cuml.testing.utils import array_equal
from cuml.datasets.classification import make_classification
from cuml.internals.safe_imports import gpu_only_import
import pytest
from functools import partial
from cuml.internals.safe_imports import cpu_only_import

np = cpu_only_import("numpy")
cp = gpu_only_import("cupy")


@pytest.mark.parametrize("n_samples", [500, 1000])
@pytest.mark.parametrize("n_features", [50, 100])
@pytest.mark.parametrize("hypercube", [True, False])
@pytest.mark.parametrize("n_classes", [2, 4])
@pytest.mark.parametrize("n_clusters_per_class", [2, 4])
@pytest.mark.parametrize("n_informative", [7, 20])
@pytest.mark.parametrize("random_state", [None, 1234])
@pytest.mark.parametrize("order", ["C", "F"])
def test_make_classification(
    n_samples,
    n_features,
    hypercube,
    n_classes,
    n_clusters_per_class,
    n_informative,
    random_state,
    order,
):

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        hypercube=hypercube,
        n_clusters_per_class=n_clusters_per_class,
        n_informative=n_informative,
        random_state=random_state,
        order=order,
    )

    assert X.shape == (n_samples, n_features)
    import cupy as cp

    assert len(cp.unique(y)) == n_classes
    assert y.shape == (n_samples,)
    if order == "F":
        assert X.flags["F_CONTIGUOUS"]
    elif order == "C":
        assert X.flags["C_CONTIGUOUS"]


def test_make_classification_informative_features():
    """Test the construction of informative features in make_classification
    Also tests `n_clusters_per_class`, `n_classes`, `hypercube` and
    fully-specified `weights`.
    """
    # Create very separate clusters; check that vertices are unique and
    # correspond to classes
    class_sep = 1e6
    make = partial(
        make_classification,
        class_sep=class_sep,
        n_redundant=0,
        n_repeated=0,
        flip_y=0,
        shift=0,
        scale=1,
        shuffle=False,
    )

    for n_informative, weights, n_clusters_per_class in [
        (2, [1], 1),
        (2, [1 / 3] * 3, 1),
        (2, [1 / 4] * 4, 1),
        (2, [1 / 2] * 2, 2),
        (2, [3 / 4, 1 / 4], 2),
        (10, [1 / 3] * 3, 10),
        (int(64), [1], 1),
    ]:
        n_classes = len(weights)
        n_clusters = n_classes * n_clusters_per_class
        n_samples = n_clusters * 50

        for hypercube in (False, True):
            X, y = make(
                n_samples=n_samples,
                n_classes=n_classes,
                weights=weights,
                n_features=n_informative,
                n_informative=n_informative,
                n_clusters_per_class=n_clusters_per_class,
                hypercube=hypercube,
                random_state=0,
            )

            assert X.shape == (n_samples, n_informative)
            assert y.shape == (n_samples,)

            # Cluster by sign, viewed as strings to allow uniquing
            signs = np.sign(cp.asnumpy(X))
            signs = signs.view(dtype="|S{0}".format(signs.strides[0])).ravel()
            unique_signs, cluster_index = np.unique(signs, return_inverse=True)

            assert (
                len(unique_signs) == n_clusters
            ), "Wrong number of clusters, or not in distinct quadrants"

            # Ensure on vertices of hypercube
            for cluster in range(len(unique_signs)):
                centroid = X[cluster_index == cluster].mean(axis=0)
                if hypercube:
                    assert array_equal(
                        cp.abs(centroid) / class_sep,
                        cp.ones(n_informative),
                        1e-5,
                    )
                else:
                    with pytest.raises(AssertionError):
                        assert array_equal(
                            cp.abs(centroid) / class_sep,
                            cp.ones(n_informative),
                            1e-5,
                        )

    with pytest.raises(ValueError):
        make(
            n_features=2, n_informative=2, n_classes=5, n_clusters_per_class=1
        )
    with pytest.raises(ValueError):
        make(
            n_features=2, n_informative=2, n_classes=3, n_clusters_per_class=2
        )

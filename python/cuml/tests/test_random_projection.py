# Copyright (c) 2018-2023, NVIDIA CORPORATION.
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

from cuml.common import has_scipy
from sklearn.datasets import make_blobs
from sklearn.random_projection import (
    johnson_lindenstrauss_min_dim as sklearn_johnson_lindenstrauss_min_dim,
)
from cuml.random_projection import (
    johnson_lindenstrauss_min_dim as cuml_johnson_lindenstrauss_min_dim,
)
from cuml.random_projection import (
    GaussianRandomProjection,
    SparseRandomProjection,
)
import pytest
from cuml.internals.safe_imports import cpu_only_import

np = cpu_only_import("numpy")


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("method", ["gaussian", "sparse"])
def test_random_projection_fit(datatype, method):
    # dataset generation
    data, target = make_blobs(n_samples=800, centers=400, n_features=3000)

    # conversion to input_type
    data = data.astype(datatype)
    target = target.astype(datatype)

    # creation of model
    if method == "gaussian":
        model = GaussianRandomProjection(eps=0.2)
    else:
        model = SparseRandomProjection(eps=0.2)

    # fitting the model
    model.fit(data)

    assert True  # Did not crash


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("method", ["gaussian", "sparse"])
def test_random_projection_fit_transform(datatype, method):
    if has_scipy():
        from scipy.spatial.distance import pdist
    else:
        pytest.skip(
            "Skipping test_random_projection_fit_transform because "
            + "Scipy is missing"
        )

    eps = 0.2

    # dataset generation
    data, target = make_blobs(n_samples=800, centers=400, n_features=3000)

    # conversion to input_type
    data = data.astype(datatype)
    target = target.astype(datatype)

    # creation of model
    if method == "gaussian":
        model = GaussianRandomProjection(eps=eps)
    else:
        model = SparseRandomProjection(eps=eps)

    # fitting the model
    model.fit(data)
    # applying transformation
    transformed_data = model.transform(data)

    original_pdist = pdist(data)
    embedded_pdist = pdist(transformed_data)

    # check JL lemma
    assert np.all(((1.0 - eps) * original_pdist) <= embedded_pdist) and np.all(
        embedded_pdist <= ((1.0 + eps) * original_pdist)
    )


def test_johnson_lindenstrauss_min_dim():
    n_tests = 10000
    n_samples = np.random.randint(low=50, high=1e10, size=n_tests)
    eps_values = np.random.rand(n_tests) + 1e-17  # range (0,1)
    tests = zip(n_samples, eps_values)

    for n_samples, eps in tests:
        cuml_value = cuml_johnson_lindenstrauss_min_dim(n_samples, eps=eps)
        sklearn_value = sklearn_johnson_lindenstrauss_min_dim(
            n_samples, eps=eps
        )
        assert cuml_value == sklearn_value


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("method", ["sparse"])
def test_random_projection_fit_transform_default(datatype, method):
    if has_scipy():
        from scipy.spatial.distance import pdist
    else:
        pytest.skip(
            "Skipping test_random_projection_fit_transform_default "
            + "because Scipy is missing"
        )

    eps = 0.8
    # dataset generation
    data, target = make_blobs(n_samples=30, centers=4, n_features=5000)

    # conversion to input_type
    data = data.astype(datatype)
    target = target.astype(datatype)

    # creation of model
    if method == "gaussian":
        model = GaussianRandomProjection()
    else:
        model = SparseRandomProjection()

    # fitting the model
    model.fit(data)
    transformed_data = model.transform(data)

    original_pdist = pdist(data)
    embedded_pdist = pdist(transformed_data)

    # check JL lemma
    assert np.all(((1.0 - eps) * original_pdist) <= embedded_pdist) and np.all(
        embedded_pdist <= ((1.0 + eps) * original_pdist)
    )

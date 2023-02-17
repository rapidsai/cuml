# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

from sklearn.utils import check_random_state
from sklearn.decomposition import TruncatedSVD as skTSVD
from sklearn.datasets import make_blobs
from cuml.testing.utils import (
    array_equal,
    unit_param,
    quality_param,
    stress_param,
)
from cuml.testing.utils import get_handle
from cuml import TruncatedSVD as cuTSVD
import pytest
from cuml.internals.safe_imports import cpu_only_import

np = cpu_only_import("numpy")


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("use_handle", [True, False])
@pytest.mark.parametrize(
    "name", [unit_param(None), quality_param("random"), stress_param("blobs")]
)
def test_tsvd_fit(datatype, name, use_handle):

    if name == "blobs":
        X, y = make_blobs(n_samples=500000, n_features=1000, random_state=0)

    elif name == "random":
        pytest.skip(
            "fails when using random dataset " "used by sklearn for testing"
        )
        shape = 5000, 100
        rng = check_random_state(42)
        X = rng.randint(-100, 20, np.product(shape)).reshape(shape)

    else:
        n, p = 500, 5
        rng = np.random.RandomState(0)
        X = rng.randn(n, p) * 0.1 + np.array([3, 4, 2, 3, 5])

    if name != "blobs":
        sktsvd = skTSVD(n_components=1)
        sktsvd.fit(X)

    handle, stream = get_handle(use_handle)
    cutsvd = cuTSVD(n_components=1, handle=handle)

    cutsvd.fit(X)
    cutsvd.handle.sync()

    if name != "blobs":
        for attr in [
            "singular_values_",
            "components_",
            "explained_variance_ratio_",
        ]:
            with_sign = False if attr in ["components_"] else True
            assert array_equal(
                getattr(cutsvd, attr),
                getattr(sktsvd, attr),
                0.4,
                with_sign=with_sign,
            )


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("use_handle", [True, False])
@pytest.mark.parametrize(
    "name", [unit_param(None), quality_param("random"), stress_param("blobs")]
)
def test_tsvd_fit_transform(datatype, name, use_handle):
    if name == "blobs":
        X, y = make_blobs(n_samples=500000, n_features=1000, random_state=0)

    elif name == "random":
        pytest.skip(
            "fails when using random dataset " "used by sklearn for testing"
        )
        shape = 5000, 100
        rng = check_random_state(42)
        X = rng.randint(-100, 20, np.product(shape)).reshape(shape)

    else:
        n, p = 500, 5
        rng = np.random.RandomState(0)
        X = rng.randn(n, p) * 0.1 + np.array([3, 4, 2, 3, 5])

    if name != "blobs":
        skpca = skTSVD(n_components=1)
        Xsktsvd = skpca.fit_transform(X)

    handle, stream = get_handle(use_handle)
    cutsvd = cuTSVD(n_components=1, handle=handle)

    Xcutsvd = cutsvd.fit_transform(X)
    cutsvd.handle.sync()

    if name != "blobs":
        assert array_equal(Xcutsvd, Xsktsvd, 1e-3, with_sign=True)


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("use_handle", [True, False])
@pytest.mark.parametrize(
    "name", [unit_param(None), quality_param("random"), stress_param("blobs")]
)
def test_tsvd_inverse_transform(datatype, name, use_handle):

    if name == "blobs":
        pytest.skip("fails when using blobs dataset")
        X, y = make_blobs(n_samples=500000, n_features=1000, random_state=0)

    elif name == "random":
        pytest.skip(
            "fails when using random dataset " "used by sklearn for testing"
        )
        shape = 5000, 100
        rng = check_random_state(42)
        X = rng.randint(-100, 20, np.product(shape)).reshape(shape)

    else:
        n, p = 500, 5
        rng = np.random.RandomState(0)
        X = rng.randn(n, p) * 0.1 + np.array([3, 4, 2, 3, 5])

    cutsvd = cuTSVD(n_components=1)
    Xcutsvd = cutsvd.fit_transform(X)
    input_gdf = cutsvd.inverse_transform(Xcutsvd)

    cutsvd.handle.sync()
    assert array_equal(input_gdf, X, 0.4, with_sign=True)

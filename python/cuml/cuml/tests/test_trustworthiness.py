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

from cuml.internals.safe_imports import cpu_only_import
import platform
import pytest
from sklearn.manifold import trustworthiness as sklearn_trustworthiness
from cuml.metrics import trustworthiness as cuml_trustworthiness

from sklearn.datasets import make_blobs

from cuml.internals.safe_imports import gpu_only_import

cudf = gpu_only_import("cudf")
np = cpu_only_import("numpy")


IS_ARM = platform.processor() == "aarch64"

if not IS_ARM:
    from umap import UMAP


@pytest.mark.parametrize("input_type", ["ndarray", "dataframe"])
@pytest.mark.parametrize("n_samples", [150, 500])
@pytest.mark.parametrize("n_features", [10, 100])
@pytest.mark.parametrize("n_components", [2, 8])
@pytest.mark.parametrize("batch_size", [128, 1024])
@pytest.mark.skipif(
    IS_ARM, reason="https://github.com/rapidsai/cuml/issues/5441"
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

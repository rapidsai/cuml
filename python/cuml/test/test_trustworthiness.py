# Copyright (c) 2018-2019, NVIDIA CORPORATION.
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

import pytest
from sklearn.manifold.t_sne import trustworthiness as sklearn_trustworthiness
from cuml.metrics import trustworthiness as cuml_trustworthiness

from sklearn.datasets import make_blobs
from umap import UMAP

import cudf
import numba.cuda
import numpy as np


@pytest.mark.parametrize('input_type', ['ndarray', 'dataframe'])
@pytest.mark.parametrize('n_samples', [10, 500])
@pytest.mark.parametrize('batch_size', [512, 2])
@pytest.mark.parametrize('n_features', [10, 100])
@pytest.mark.parametrize('n_components', [2, 8])
def test_trustworthiness(input_type, n_samples, n_features, n_components,
                         batch_size):
    centers = round(n_samples*0.4)
    X, y = make_blobs(n_samples=n_samples, centers=centers,
                      n_features=n_features, random_state=32)

    X_embedded = \
        UMAP(n_components=n_components, random_state=32).fit_transform(X)
    X = X.astype(np.float32)
    X_embedded = X_embedded.astype(np.float32)

    sk_score = sklearn_trustworthiness(X, X_embedded)

    if input_type == 'dataframe':
        X = cudf.DataFrame.from_gpu_matrix(
            numba.cuda.to_device(X))

        X_embedded = cudf.DataFrame.from_gpu_matrix(
            numba.cuda.to_device(X_embedded))

    score = cuml_trustworthiness(X, X_embedded, batch_size=batch_size)

    assert abs(score - sk_score) <= 1e-3

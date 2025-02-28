#
# Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
import pytest

import numpy as np
import cuml
from cuml.datasets import make_blobs

pytestmark = pytest.mark.filterwarnings(
    "ignore:The default value of `n_init` will change from 1 to 'auto' in 25.04"
)


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
def test_random_state_argument(Estimator):
    X, y = make_blobs(random_state=0)
    # Check that both integer and np.random.RandomState are accepted
    for seed in (42, np.random.RandomState(42)):
        est = Estimator(random_state=seed)

        if est.__class__.__name__ != "TSNE":
            est.fit(X, y)
        else:
            est.fit(X)

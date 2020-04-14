#
# Copyright (c) 2020, NVIDIA CORPORATION.
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
from cuml.datasets.classification import make_classification


@pytest.mark.parametrize('n_samples', [1000])
@pytest.mark.parametrize('n_features', [100])
@pytest.mark.parametrize('n_classes', [2, 4])
@pytest.mark.parametrize('n_clusters_per_class', [2, 4])
@pytest.mark.parametrize('n_informative', [7])
@pytest.mark.parametrize('random_state', [None, 1234])
@pytest.mark.parametrize('order', ['C', 'F'])
def test_make_classification(n_samples, n_features, n_classes,
                             n_clusters_per_class, n_informative,
                             random_state, order):

    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_classes=n_classes,
                               n_clusters_per_class=n_clusters_per_class,
                               n_informative=n_informative,
                               random_state=random_state, order=order)

    assert X.shape == (n_samples, n_features)
    import cupy as cp
    assert len(cp.unique(y)) == n_classes
    assert y.shape == (n_samples, )
    if order == 'F':
        assert X.flags['F_CONTIGUOUS']
    elif order == 'C':
        assert X.flags['C_CONTIGUOUS']

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
import cupy as cp

from sklearn.decomposition import IncrementalPCA as skIPCA

from cuml.datasets import make_blobs
from cuml.decomposition import IncrementalPCA as cuIPCA

from cuml.test.utils import array_equal


@pytest.mark.parametrize('nrows', [500, 5000])
@pytest.mark.parametrize('ncols', [15, 25])
@pytest.mark.parametrize('n_components', [10, 12])
@pytest.mark.parametrize('sparse_input', [True, False])
def test_inverse_transform(nrows, ncols, n_components, sparse_input):

    if sparse_input:
        X = cp.sparse.random(nrows, ncols, density=0.07,
                             random_state=10).tocsr()
    else:
        X, _ = make_blobs(n_samples=nrows, n_features=ncols, random_state=10)

    cu_ipca = cuIPCA(n_components=n_components)
    cu_ipca.fit(X)
    cu_t = cu_ipca.transform(X)
    cu_inv = cu_ipca.inverse_transform(cu_t)

    sk_ipca = skIPCA(n_components=n_components)
    if sparse_input:
        X = X.get()
    else:
        X = cp.asnumpy(X)
    sk_ipca.fit(X)
    sk_t = sk_ipca.transform(X)
    sk_inv = sk_ipca.inverse_transform(sk_t)

    assert array_equal(cu_inv, sk_inv,
                       5e-5, with_sign=True)

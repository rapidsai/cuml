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
import cupyx

from sklearn.decomposition import IncrementalPCA as skIPCA

from cuml.datasets import make_blobs
from cuml.experimental.decomposition import IncrementalPCA as cuIPCA

from cuml.test.utils import array_equal


@pytest.mark.parametrize(
    'nrows, ncols, n_components, sparse_input, density, sparse_format,'
    ' batch_size_divider', [
        (500, 15, 2, True, 0.4, 'csr', 5),
        (5000, 25, 12, False, 0.07, 'csc', 10),
        (5000, 15, None, True, 0.4, 'csc', 5),
        (500, 25, 2, False, 0.07, 'csr', 10),
        (5000, 25, 12, False, 0.07, 'csr', 10)
    ]
)
@pytest.mark.no_bad_cuml_array_check
def test_fit(nrows, ncols, n_components, sparse_input, density,
             sparse_format, batch_size_divider):

    if sparse_format == 'csc':
        pytest.skip("cupyx.scipy.sparse.csc.csc_matrix does not support"
                    " indexing as of cupy 7.6.0")

    if sparse_input:
        X = cupyx.scipy.sparse.random(nrows, ncols, density=density,
                                      random_state=10, format=sparse_format)
    else:
        X, _ = make_blobs(n_samples=nrows, n_features=ncols, random_state=10)

    cu_ipca = cuIPCA(n_components=n_components,
                     batch_size=int(nrows / batch_size_divider))
    cu_ipca.fit(X)
    cu_t = cu_ipca.transform(X)
    cu_inv = cu_ipca.inverse_transform(cu_t)

    sk_ipca = skIPCA(n_components=n_components,
                     batch_size=int(nrows / batch_size_divider))
    if sparse_input:
        X = X.get()
    else:
        X = cp.asnumpy(X)
    sk_ipca.fit(X)
    sk_t = sk_ipca.transform(X)
    sk_inv = sk_ipca.inverse_transform(sk_t)

    assert array_equal(cu_inv, sk_inv,
                       5e-5, with_sign=True)


@pytest.mark.parametrize(
    'nrows, ncols, n_components, density, batch_size_divider', [
        (500, 15, 2, 0.07, 5),
        (5000, 25, 12, 0.07, 10),
        (5000, 15, 2, 0.4, 5),
        (500, 25, 12, 0.4, 10),
    ]
)
@pytest.mark.no_bad_cuml_array_check
def test_partial_fit(nrows, ncols, n_components, density,
                     batch_size_divider):

    X, _ = make_blobs(n_samples=nrows, n_features=ncols, random_state=10)

    cu_ipca = cuIPCA(n_components=n_components)

    sample_size = int(nrows / batch_size_divider)
    for i in range(0, nrows, sample_size):
        cu_ipca.partial_fit(X[i:i + sample_size].copy())

    cu_t = cu_ipca.transform(X)
    cu_inv = cu_ipca.inverse_transform(cu_t)

    sk_ipca = skIPCA(n_components=n_components)

    X = cp.asnumpy(X)

    for i in range(0, nrows, sample_size):
        sk_ipca.partial_fit(X[i:i + sample_size].copy())

    sk_t = sk_ipca.transform(X)
    sk_inv = sk_ipca.inverse_transform(sk_t)

    assert array_equal(cu_inv, sk_inv,
                       5e-5, with_sign=True)

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

import pytest
from dask_cuda import LocalCUDACluster

from dask.distributed import Client, wait

import numpy as np


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [6e5])
@pytest.mark.parametrize("ncols", [20])
@pytest.mark.parametrize("n_parts", [67])
def test_pca_fit(nrows, ncols, n_parts, client=None):

    owns_cluster = False
    if client is None:
        owns_cluster = True
        cluster = LocalCUDACluster(threads_per_worker=1)
        client = Client(cluster)

    from cuml.dask.decomposition import TruncatedSVD as daskTPCA
    from sklearn.decomposition import TruncatedSVD

    from cuml.dask.datasets import make_blobs

    X_cudf, _ = make_blobs(nrows, ncols, 1, n_parts,
                           cluster_std=0.5, verbose=False,
                           random_state=10, dtype=np.float32)

    wait(X_cudf)

    X = X_cudf.compute().to_pandas().values

    cutsvd = daskTPCA(n_components=5)
    cutsvd.fit(X_cudf)

    sktsvd = TruncatedSVD(n_components=5, algorithm="arpack")
    sktsvd.fit(X)

    from cuml.test.utils import array_equal

    all_attr = ['singular_values_', 'components_',
            'explained_variance_', 'explained_variance_ratio_']

    if owns_cluster:
        client.close()
        cluster.close()

    for attr in all_attr:
        with_sign = False if attr in ['components_'] else True
        cuml_res = (getattr(cutsvd, attr))
        if type(cuml_res) == np.ndarray:
            cuml_res = cuml_res.as_matrix()
        skl_res = getattr(sktsvd, attr)
        if attr == 'singular_values_':
            assert array_equal(cuml_res, skl_res, 1, with_sign=with_sign)
        else:
            assert array_equal(cuml_res, skl_res, 1e-2, with_sign=with_sign)


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [4e3, 7e5])
@pytest.mark.parametrize("ncols", [100, 1000])
@pytest.mark.parametrize("n_parts", [46])
def test_pca_fit_transform_fp32(nrows, ncols, n_parts, client=None):

    owns_cluster = False
    if client is None:
        owns_cluster = True
        cluster = LocalCUDACluster(threads_per_worker=1)
        client = Client(cluster)

    from cuml.dask.decomposition import TruncatedSVD as daskTPCA
    from cuml.dask.datasets import make_blobs

    X_cudf, _ = make_blobs(nrows, ncols, 1, n_parts,
                           cluster_std=1.5, verbose=False,
                           random_state=10, dtype=np.float32)

    wait(X_cudf)

    cutsvd = daskTPCA(n_components=20)
    cutsvd.fit_transform(X_cudf)

    if owns_cluster:
        client.close()
        cluster.close()


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [7e5])
@pytest.mark.parametrize("ncols", [200])
@pytest.mark.parametrize("n_parts", [33])
def test_pca_fit_transform_fp64(nrows, ncols, n_parts, client=None):

    owns_cluster = False
    if client is None:
        owns_cluster = True
        cluster = LocalCUDACluster(threads_per_worker=1)
        client = Client(cluster)

    from cuml.dask.decomposition import TruncatedSVD as daskTPCA
    from cuml.dask.datasets import make_blobs

    X_cudf, _ = make_blobs(nrows, ncols, 1, n_parts,
                           cluster_std=1.5, verbose=False,
                           random_state=10, dtype=np.float64)

    wait(X_cudf)

    cutsvd = daskTPCA(n_components=30)
    cutsvd.fit_transform(X_cudf)

    if owns_cluster:
        client.close()
        cluster.close()

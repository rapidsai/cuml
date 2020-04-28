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
from dask.distributed import Client, wait

import numpy as np
from cuml.test.utils import array_equal, \
    unit_param, stress_param


@pytest.mark.mg
@pytest.mark.parametrize("data_info", [unit_param([1000, 20, 30]),
                         stress_param([9e6, 5000, 30])])
def test_pca_fit(data_info, cluster):

    client = Client(cluster)
    nrows, ncols, n_parts = data_info

    try:

        from cuml.dask.decomposition import TruncatedSVD as daskTPCA
        from sklearn.decomposition import TruncatedSVD

        from cuml.dask.datasets import make_blobs

        X_cudf, _ = make_blobs(n_samples=nrows,
                               n_features=ncols,
                               centers=1,
                               n_parts=n_parts,
                               cluster_std=0.5, verbose=False,
                               random_state=10, dtype=np.float32)

        wait(X_cudf)

        X = X_cudf.compute().to_pandas().values

        cutsvd = daskTPCA(n_components=5)
        cutsvd.fit(X_cudf)

        sktsvd = TruncatedSVD(n_components=5, algorithm="arpack")
        sktsvd.fit(X)

        all_attr = ['singular_values_', 'components_',
                    'explained_variance_', 'explained_variance_ratio_']

    finally:
        client.close()

    for attr in all_attr:
        with_sign = False if attr in ['components_'] else True
        cuml_res = (getattr(cutsvd, attr))
        if type(cuml_res) == np.ndarray:
            cuml_res = cuml_res.as_matrix()
        skl_res = getattr(sktsvd, attr)
        if attr == 'singular_values_':
            assert array_equal(cuml_res, skl_res, 1, with_sign=with_sign)
        else:
            assert array_equal(cuml_res, skl_res, 1e-1, with_sign=with_sign)


@pytest.mark.mg
@pytest.mark.parametrize("data_info", [unit_param([1000, 20, 46]),
                         stress_param([9e6, 5000, 46])])
def test_pca_fit_transform_fp32(data_info, cluster):

    client = Client(cluster)
    nrows, ncols, n_parts = data_info
    try:
        from cuml.dask.decomposition import TruncatedSVD as daskTPCA
        from cuml.dask.datasets import make_blobs

        X_cudf, _ = make_blobs(n_samples=nrows,
                               n_features=ncols,
                               centers=1,
                               n_parts=n_parts,
                               cluster_std=1.5, verbose=False,
                               random_state=10, dtype=np.float32)

        wait(X_cudf)

        cutsvd = daskTPCA(n_components=20)
        cutsvd.fit_transform(X_cudf)

    finally:
        client.close()


@pytest.mark.mg
@pytest.mark.parametrize("data_info", [unit_param([1000, 20, 33]),
                         stress_param([9e6, 5000, 33])])
def test_pca_fit_transform_fp64(data_info, cluster):

    client = Client(cluster)
    nrows, ncols, n_parts = data_info

    try:
        from cuml.dask.decomposition import TruncatedSVD as daskTPCA
        from cuml.dask.datasets import make_blobs

        X_cudf, _ = make_blobs(n_samples=nrows,
                               n_features=ncols,
                               centers=1,
                               n_parts=n_parts,
                               cluster_std=1.5, verbose=False,
                               random_state=10, dtype=np.float64)

        wait(X_cudf)

        cutsvd = daskTPCA(n_components=30)
        cutsvd.fit_transform(X_cudf)

    finally:
        client.close()

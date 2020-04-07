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
@pytest.mark.parametrize("nrows", [unit_param(6e5),
                         stress_param(5e6)])
@pytest.mark.parametrize("ncols", [unit_param(20),
                         stress_param(1000)])
@pytest.mark.parametrize("n_parts", [unit_param(67)])
def test_pca_fit(nrows, ncols, n_parts, cluster):

    client = Client(cluster)

    try:

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
@pytest.mark.parametrize("nrows", [unit_param(4e3),
                         unit_param(7e5),
                         stress_param(9e6)])
@pytest.mark.parametrize("ncols", [unit_param(100),
                         unit_param(1000),
                         stress_param(5000)])
@pytest.mark.parametrize("n_parts", [46])
def test_pca_fit_transform_fp32(nrows, ncols, n_parts, cluster):

    client = Client(cluster)

    try:
        from cuml.dask.decomposition import TruncatedSVD as daskTPCA
        from cuml.dask.datasets import make_blobs

        X_cudf, _ = make_blobs(nrows, ncols, 1, n_parts,
                               cluster_std=1.5, verbose=False,
                               random_state=10, dtype=np.float32)

        wait(X_cudf)

        cutsvd = daskTPCA(n_components=20)
        cutsvd.fit_transform(X_cudf)

    finally:
        client.close()


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [unit_param(7e5),
                         stress_param(9e6)])
@pytest.mark.parametrize("ncols", [unit_param(200),
                         stress_param(5000)])
@pytest.mark.parametrize("n_parts", [unit_param(33)])
def test_pca_fit_transform_fp64(nrows, ncols, n_parts, cluster):

    client = Client(cluster)

    try:
        from cuml.dask.decomposition import TruncatedSVD as daskTPCA
        from cuml.dask.datasets import make_blobs

        X_cudf, _ = make_blobs(nrows, ncols, 1, n_parts,
                               cluster_std=1.5, verbose=False,
                               random_state=10, dtype=np.float64)

        wait(X_cudf)

        cutsvd = daskTPCA(n_components=30)
        cutsvd.fit_transform(X_cudf)

    finally:
        client.close()

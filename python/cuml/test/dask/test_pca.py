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


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [6e5])
@pytest.mark.parametrize("ncols", [20])
@pytest.mark.parametrize("n_parts", [67])
def test_pca_fit(nrows, ncols, n_parts, cluster):

    client = Client(cluster)

    try:

        from cuml.dask.decomposition import PCA as daskPCA
        from sklearn.decomposition import PCA

        from cuml.dask.datasets import make_blobs

        X_cudf, _ = make_blobs(nrows, ncols, 1, n_parts,
                               cluster_std=0.5, verbose=False,
                               random_state=10, dtype=np.float32)

        wait(X_cudf)

        print(str(X_cudf.head(3)))

        try:

            cupca = daskPCA(n_components=5, whiten=True)
            cupca.fit(X_cudf)
        except Exception as e:
            print(str(e))

        X = X_cudf.compute().to_pandas().values

        skpca = PCA(n_components=5, whiten=True, svd_solver="full")
        skpca.fit(X)

        from cuml.test.utils import array_equal

        all_attr = ['singular_values_', 'components_',
                    'explained_variance_', 'explained_variance_ratio_']

        for attr in all_attr:
            with_sign = False if attr in ['components_'] else True
            cuml_res = (getattr(cupca, attr))
            if type(cuml_res) == np.ndarray:
                cuml_res = cuml_res.as_matrix()
            skl_res = getattr(skpca, attr)
            assert array_equal(cuml_res, skl_res, 1e-3, with_sign=with_sign)
    finally:
        client.close()


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [4e3, 7e5])
@pytest.mark.parametrize("ncols", [100, 1000])
@pytest.mark.parametrize("n_parts", [46])
def test_pca_fit_transform_fp32(nrows, ncols, n_parts, cluster):

    client = Client(cluster)

    try:
        from cuml.dask.decomposition import PCA as daskPCA
        from cuml.dask.datasets import make_blobs

        X_cudf, _ = make_blobs(nrows, ncols, 1, n_parts,
                               cluster_std=1.5, verbose=False,
                               random_state=10, dtype=np.float32)

        wait(X_cudf)

        cupca = daskPCA(n_components=20, whiten=True)
        cupca.fit_transform(X_cudf)

    finally:
        client.close()


@pytest.mark.mg
@pytest.mark.parametrize("nrows", [7e5])
@pytest.mark.parametrize("ncols", [200])
@pytest.mark.parametrize("n_parts", [33])
def test_pca_fit_transform_fp64(nrows, ncols, n_parts, cluster):

    client = Client(cluster)

    try:
        from cuml.dask.decomposition import PCA as daskPCA
        from cuml.dask.datasets import make_blobs

        X_cudf, _ = make_blobs(nrows, ncols, 1, n_parts,
                               cluster_std=1.5, verbose=False,
                               random_state=10, dtype=np.float64)

        wait(X_cudf)

        cupca = daskPCA(n_components=30, whiten=False)
        cupca.fit_transform(X_cudf)

    finally:
        client.close()

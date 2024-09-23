# Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

from cuml.dask.common.dask_arr_utils import to_dask_cudf
from cuml.internals.safe_imports import gpu_only_import
from cuml.testing.utils import array_equal, unit_param, stress_param
import pytest

from cuml.internals.safe_imports import cpu_only_import

np = cpu_only_import("numpy")
cp = gpu_only_import("cupy")


@pytest.mark.mg
@pytest.mark.parametrize(
    "data_info",
    [unit_param([1000, 20, 30]), stress_param([int(9e6), 5000, 30])],
)
@pytest.mark.parametrize("input_type", ["dataframe", "array"])
def test_pca_fit(data_info, input_type, client):

    nrows, ncols, n_parts = data_info
    if nrows == int(9e6) and pytest.max_gpu_memory < 48:
        if pytest.adapt_stress_test:
            nrows = nrows * pytest.max_gpu_memory // 256
            ncols = ncols * pytest.max_gpu_memory // 256
        else:
            pytest.skip(
                "Insufficient GPU memory for this test."
                "Re-run with 'CUML_ADAPT_STRESS_TESTS=True'"
            )

    from cuml.dask.decomposition import TruncatedSVD as daskTPCA
    from sklearn.decomposition import TruncatedSVD

    from cuml.dask.datasets import make_blobs

    X, _ = make_blobs(
        n_samples=nrows,
        n_features=ncols,
        centers=1,
        n_parts=n_parts,
        cluster_std=0.5,
        random_state=10,
        dtype=np.float32,
    )

    if input_type == "dataframe":
        X_train = to_dask_cudf(X)
        X_cpu = X_train.compute().to_pandas().values
    elif input_type == "array":
        X_train = X
        X_cpu = cp.asnumpy(X_train.compute())

    cutsvd = daskTPCA(n_components=5)
    cutsvd.fit(X_train)

    sktsvd = TruncatedSVD(n_components=5, algorithm="arpack")
    sktsvd.fit(X_cpu)

    all_attr = [
        "singular_values_",
        "components_",
        "explained_variance_",
        "explained_variance_ratio_",
    ]

    for attr in all_attr:
        with_sign = False if attr in ["components_"] else True
        cuml_res = getattr(cutsvd, attr)
        if type(cuml_res) is np.ndarray:
            cuml_res = cuml_res.to_numpy()
        skl_res = getattr(sktsvd, attr)
        if attr == "singular_values_":
            assert array_equal(cuml_res, skl_res, 1, with_sign=with_sign)
        else:
            assert array_equal(cuml_res, skl_res, 1e-1, with_sign=with_sign)


@pytest.mark.mg
@pytest.mark.parametrize(
    "data_info",
    [unit_param([1000, 20, 46]), stress_param([int(9e6), 5000, 46])],
)
def test_pca_fit_transform_fp32(data_info, client):

    nrows, ncols, n_parts = data_info
    from cuml.dask.decomposition import TruncatedSVD as daskTPCA
    from cuml.dask.datasets import make_blobs

    X_cudf, _ = make_blobs(
        n_samples=nrows,
        n_features=ncols,
        centers=1,
        n_parts=n_parts,
        cluster_std=1.5,
        random_state=10,
        dtype=np.float32,
    )

    cutsvd = daskTPCA(n_components=20)
    cutsvd.fit_transform(X_cudf)


@pytest.mark.mg
@pytest.mark.parametrize(
    "data_info",
    [unit_param([1000, 20, 33]), stress_param([int(9e6), 5000, 33])],
)
def test_pca_fit_transform_fp64(data_info, client):

    nrows, ncols, n_parts = data_info

    from cuml.dask.decomposition import TruncatedSVD as daskTPCA
    from cuml.dask.datasets import make_blobs

    X_cudf, _ = make_blobs(
        n_samples=nrows,
        n_features=ncols,
        centers=1,
        n_parts=n_parts,
        cluster_std=1.5,
        random_state=10,
        dtype=np.float64,
    )

    cutsvd = daskTPCA(n_components=30)
    cutsvd.fit_transform(X_cudf)

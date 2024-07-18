# Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

from cuml.testing.test_preproc_utils import assert_allclose
from sklearn.utils.sparsefuncs import (
    inplace_csr_column_scale as sk_inplace_csr_column_scale,
    inplace_csr_row_scale as sk_inplace_csr_row_scale,
    inplace_column_scale as sk_inplace_column_scale,
    mean_variance_axis as sk_mean_variance_axis,
    min_max_axis as sk_min_max_axis,
)
from cuml._thirdparty.sklearn.utils.sparsefuncs import (
    inplace_csr_column_scale as cu_inplace_csr_column_scale,
    inplace_csr_row_scale as cu_inplace_csr_row_scale,
    inplace_column_scale as cu_inplace_column_scale,
    mean_variance_axis as cu_mean_variance_axis,
    min_max_axis as cu_min_max_axis,
)
from sklearn.utils.extmath import (
    row_norms as sk_row_norms,
    _incremental_mean_and_var as sk_incremental_mean_and_var,
)
from cuml._thirdparty.sklearn.utils.extmath import (
    row_norms as cu_row_norms,
    _incremental_mean_and_var as cu_incremental_mean_and_var,
)
from cuml._thirdparty.sklearn.utils.validation import check_X_y
from cuml.internals.safe_imports import gpu_only_import
import pytest

from cuml.internals.safe_imports import cpu_only_import

np = cpu_only_import("numpy")
cp = gpu_only_import("cupy")
cpx = gpu_only_import("cupyx")


@pytest.fixture(scope="session")
def random_dataset(request, random_seed):
    cp.random.seed(random_seed)
    X = cp.random.rand(100, 10)
    return X.get(), X


@pytest.fixture(scope="session", params=["cupy-csr", "cupy-csc"])
def sparse_random_dataset(request, random_seed):
    cp.random.seed(random_seed)
    X = cp.random.rand(100, 10)
    random_loc = cp.random.choice(X.size, int(X.size * 0.3), replace=False)
    X.ravel()[random_loc] = 0
    if request.param == "cupy-csr":
        X_sparse = cpx.scipy.sparse.csr_matrix(X)
    elif request.param == "cupy-csc":
        X_sparse = cpx.scipy.sparse.csc_matrix(X)
    return X.get(), X, X_sparse.get(), X_sparse


def test_check_X_y():
    X = np.ones((100, 10))
    y1 = np.ones((100,))
    y2 = np.ones((100, 1))
    y3 = np.ones((100, 2))
    y4 = np.ones((101,))

    check_X_y(X, y1, multi_output=False)
    check_X_y(X, y2, multi_output=False)
    with pytest.raises(Exception):
        check_X_y(X, y3, multi_output=False)
    with pytest.raises(Exception):
        check_X_y(X, y4, multi_output=False)
    with pytest.raises(Exception):
        check_X_y(X, y4, multi_output=True)


@pytest.mark.parametrize("square", [False, True])
def test_row_norms(failure_logger, sparse_random_dataset, square):
    X_np, X, X_sparse_np, X_sparse = sparse_random_dataset

    cu_norms = cu_row_norms(X, squared=square)
    sk_norms = sk_row_norms(X_np, squared=square)
    assert_allclose(cu_norms, sk_norms)

    cu_norms = cu_row_norms(X_sparse, squared=square)
    sk_norms = sk_row_norms(X_sparse_np, squared=square)
    assert_allclose(cu_norms, sk_norms)


def test_incremental_mean_and_var(failure_logger, random_seed, random_dataset):
    X_np, X = random_dataset
    cp.random.seed(random_seed)
    last_mean = cp.random.rand(10)
    last_variance = cp.random.rand(10)
    last_sample_count = cp.random.rand(10)

    cu_mean, cu_variance, cu_sample_count = cu_incremental_mean_and_var(
        X, last_mean, last_variance, last_sample_count
    )
    sk_mean, sk_variance, sk_sample_count = sk_incremental_mean_and_var(
        X_np, last_mean.get(), last_variance.get(), last_sample_count.get()
    )
    assert_allclose(cu_mean, sk_mean)
    assert_allclose(cu_variance, sk_variance)
    assert_allclose(cu_sample_count, sk_sample_count)


def test_inplace_csr_column_scale(
    failure_logger, random_seed, sparse_random_dataset
):
    _, _, X_sparse_np, X_sparse = sparse_random_dataset
    if X_sparse.format != "csr":
        pytest.skip()
    cp.random.seed(random_seed)
    scale = cp.random.rand(10)
    cu_inplace_csr_column_scale(X_sparse, scale)
    sk_inplace_csr_column_scale(X_sparse_np, scale.get())
    assert_allclose(X_sparse, X_sparse_np)


def test_inplace_csr_row_scale(
    failure_logger, random_seed, sparse_random_dataset
):
    _, _, X_sparse_np, X_sparse = sparse_random_dataset
    if X_sparse.format != "csr":
        pytest.skip()
    cp.random.seed(random_seed)
    scale = cp.random.rand(100)
    cu_inplace_csr_row_scale(X_sparse, scale)
    sk_inplace_csr_row_scale(X_sparse_np, scale.get())
    assert_allclose(X_sparse, X_sparse_np)


def test_inplace_column_scale(
    failure_logger, random_seed, sparse_random_dataset
):
    _, X, X_sparse_np, X_sparse = sparse_random_dataset
    cp.random.seed(random_seed)
    scale = cp.random.rand(10)
    cu_inplace_column_scale(X_sparse, scale)
    sk_inplace_column_scale(X_sparse_np, scale.get())
    assert_allclose(X_sparse, X_sparse_np)
    with pytest.raises(Exception):
        cu_inplace_column_scale(X, scale)


@pytest.mark.parametrize("axis", [0, 1])
def test_mean_variance_axis(failure_logger, sparse_random_dataset, axis):
    _, _, X_sparse_np, X_sparse = sparse_random_dataset
    cu_mean, cu_variance = cu_mean_variance_axis(X_sparse, axis=axis)
    sk_mean, sk_variance = sk_mean_variance_axis(X_sparse_np, axis=axis)
    assert_allclose(cu_mean, sk_mean)
    assert_allclose(cu_variance, sk_variance)


@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize("ignore_nan", [False, True])
# ignore warning about changing sparsity in both cupy and scipy
@pytest.mark.filterwarnings("ignore:(.*)expensive(.*)::")
def test_min_max_axis(failure_logger, sparse_random_dataset, axis, ignore_nan):
    _, X, X_sparse_np, X_sparse = sparse_random_dataset
    X_sparse[0, 0] = np.nan
    X_sparse_np[0, 0] = np.nan
    cu_min, cu_max = cu_min_max_axis(
        X_sparse, axis=axis, ignore_nan=ignore_nan
    )
    sk_min, sk_max = sk_min_max_axis(
        X_sparse_np, axis=axis, ignore_nan=ignore_nan
    )

    if axis is not None:
        assert_allclose(cu_min, sk_min)
        assert_allclose(cu_max, sk_max)
    else:
        assert cu_min == sk_min or (cp.isnan(cu_min) and np.isnan(sk_min))
        assert cu_max == sk_max or (cp.isnan(cu_max) and np.isnan(sk_max))

    with pytest.raises(Exception):
        cu_min_max_axis(X, axis=axis, ignore_nan=ignore_nan)


@pytest.fixture(scope="session", params=["cupy-csr", "cupy-csc"])
def sparse_extremes(request, random_seed):
    X = cp.array(
        [
            [-0.9933658, 0.871748, 0.44418066],
            [0.87808335, cp.nan, 0.18183318],
            [cp.nan, 0.25030251, -0.7269053],
            [cp.nan, 0.17725405, cp.nan],
            [cp.nan, cp.nan, cp.nan],
            [0.0, 0.0, 0.44418066],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, cp.nan],
            [0.0, cp.nan, cp.nan],
        ]
    )
    if request.param == "cupy-csr":
        X_sparse = cpx.scipy.sparse.csr_matrix(X)
    elif request.param == "cupy-csc":
        X_sparse = cpx.scipy.sparse.csc_matrix(X)
    return X_sparse.get(), X_sparse


@pytest.mark.parametrize("axis", [None, 0, 1])
@pytest.mark.parametrize("ignore_nan", [False, True])
# ignore warning about changing sparsity in both cupy and scipy
@pytest.mark.filterwarnings("ignore:(.*)expensive(.*)::")
# ignore warning about all nan row in sparse_extremes
@pytest.mark.filterwarnings("ignore:All-NaN(.*)::")
def test_min_max_axis_extremes(sparse_extremes, axis, ignore_nan):
    X_sparse_np, X_sparse = sparse_extremes

    cu_min, cu_max = cu_min_max_axis(
        X_sparse, axis=axis, ignore_nan=ignore_nan
    )
    sk_min, sk_max = sk_min_max_axis(
        X_sparse_np, axis=axis, ignore_nan=ignore_nan
    )

    if axis is not None:
        assert_allclose(cu_min, sk_min)
        assert_allclose(cu_max, sk_max)
    else:
        assert cu_min == sk_min or (cp.isnan(cu_min) and np.isnan(sk_min))
        assert cu_max == sk_max or (cp.isnan(cu_max) and np.isnan(sk_max))

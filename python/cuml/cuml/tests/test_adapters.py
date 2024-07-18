#
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

import platform
from sklearn.preprocessing import normalize as sk_normalize
from cuml.testing.test_preproc_utils import assert_allclose
from cuml.thirdparty_adapters.sparsefuncs_fast import (
    csr_mean_variance_axis0,
    csc_mean_variance_axis0,
    _csc_mean_variance_axis0,
    inplace_csr_row_normalize_l1,
    inplace_csr_row_normalize_l2,
)
from sklearn.utils._mask import _get_mask as sk_get_mask
from cuml.thirdparty_adapters.adapters import (
    check_array,
    _get_mask as cu_get_mask,
    _masked_column_median,
    _masked_column_mean,
    _masked_column_mode,
)
from cuml.internals.safe_imports import cpu_only_import_from
from cuml.internals.safe_imports import gpu_only_import_from
from cuml.internals.safe_imports import cpu_only_import
import pytest

from cuml.internals.safe_imports import gpu_only_import

cp = gpu_only_import("cupy")
cpx = gpu_only_import("cupyx")
np = cpu_only_import("numpy")
coo_matrix = gpu_only_import_from("cupyx.scipy.sparse", "coo_matrix")
stats = cpu_only_import_from("scipy", "stats")


IS_ARM = platform.processor() == "aarch64"


@pytest.fixture(scope="session", params=["zero", "one", "nan"])
def mask_dataset(request, random_seed):
    cp.random.seed(random_seed)
    randint = cp.random.randint(30, size=(500, 20))
    randint = randint.astype(cp.float64)
    if request.param == "zero":
        mask_value = 0
    elif request.param == "one":
        mask_value = 1
    else:
        mask_value = cp.nan
    random_loc = cp.random.choice(
        randint.size, int(randint.size * 0.3), replace=False
    )
    randint.ravel()[random_loc] = mask_value
    return mask_value, randint.get(), randint


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


@pytest.mark.skipif(
    IS_ARM,
    reason="Test fails unexpectedly on ARM. "
    "github.com/rapidsai/cuml/issues/5100",
)
def test_check_array():
    # accept_sparse
    arr = coo_matrix((3, 4), dtype=cp.float64)
    check_array(arr, accept_sparse=True)
    with pytest.raises(ValueError):
        check_array(arr, accept_sparse=False)

    # dtype
    arr = cp.array([[1, 2]], dtype=cp.int64)
    check_array(arr, dtype=cp.int64, copy=False)

    arr = cp.array([[1, 2]], dtype=cp.float32)
    new_arr = check_array(arr, dtype=cp.int64)
    assert new_arr.dtype == cp.int64

    # order
    arr = cp.array([[1, 2]], dtype=cp.int64, order="F")
    new_arr = check_array(arr, order="F")
    assert new_arr.flags.f_contiguous
    new_arr = check_array(arr, order="C")
    assert new_arr.flags.c_contiguous

    # force_all_finite
    arr = cp.array([[1, cp.inf]])
    check_array(arr, force_all_finite=False)
    with pytest.raises(ValueError):
        check_array(arr, force_all_finite=True)

    # ensure_2d
    arr = cp.array([1, 2], dtype=cp.float32)
    check_array(arr, ensure_2d=False)
    with pytest.raises(ValueError):
        check_array(arr, ensure_2d=True)

    # ensure_2d
    arr = cp.array([[1, 2, 3], [4, 5, 6]], dtype=cp.float32)
    check_array(arr, ensure_2d=True)

    # ensure_min_samples
    arr = cp.array([[1, 2]], dtype=cp.float32)
    check_array(arr, ensure_min_samples=1)
    with pytest.raises(ValueError):
        check_array(arr, ensure_min_samples=2)

    # ensure_min_features
    arr = cp.array([[]], dtype=cp.float32)
    check_array(arr, ensure_min_features=0)
    with pytest.raises(ValueError):
        check_array(arr, ensure_min_features=1)


def test_csr_mean_variance_axis0(failure_logger, sparse_random_dataset):
    X_np, _, _, X_sparse = sparse_random_dataset
    if X_sparse.format != "csr":
        pytest.skip("Skip non CSR matrices")

    means, variances = csr_mean_variance_axis0(X_sparse)

    ref_means = np.nanmean(X_np, axis=0)
    ref_variances = np.nanvar(X_np, axis=0)

    assert_allclose(means, ref_means)
    assert_allclose(variances, ref_variances)


def test_csc_mean_variance_axis0(failure_logger, sparse_random_dataset):
    X_np, _, _, X_sparse = sparse_random_dataset
    if X_sparse.format != "csc":
        pytest.skip("Skip non CSC matrices")

    means, variances = csc_mean_variance_axis0(X_sparse)

    ref_means = np.nanmean(X_np, axis=0)
    ref_variances = np.nanvar(X_np, axis=0)

    assert_allclose(means, ref_means)
    assert_allclose(variances, ref_variances)


def test__csc_mean_variance_axis0(failure_logger, sparse_random_dataset):
    X_np, _, _, X_sparse = sparse_random_dataset
    if X_sparse.format != "csc":
        pytest.skip("Skip non CSC matrices")

    means, variances, counts_nan = _csc_mean_variance_axis0(X_sparse)

    ref_means = np.nanmean(X_np, axis=0)
    ref_variances = np.nanvar(X_np, axis=0)
    ref_counts_nan = np.isnan(X_np).sum(axis=0)

    assert_allclose(means, ref_means)
    assert_allclose(variances, ref_variances)
    assert_allclose(counts_nan, ref_counts_nan)


def test_inplace_csr_row_normalize_l1(failure_logger, sparse_random_dataset):
    X_np, _, _, X_sparse = sparse_random_dataset
    if X_sparse.format != "csr":
        pytest.skip("Skip non CSR matrices")

    inplace_csr_row_normalize_l1(X_sparse)
    X_np = sk_normalize(X_np, norm="l1", axis=1)
    assert_allclose(X_sparse, X_np)


def test_inplace_csr_row_normalize_l2(failure_logger, sparse_random_dataset):
    X_np, _, _, X_sparse = sparse_random_dataset
    if X_sparse.format != "csr":
        pytest.skip("Skip non CSR matrices")

    inplace_csr_row_normalize_l2(X_sparse)
    X_np = sk_normalize(X_np, norm="l2", axis=1)
    assert_allclose(X_sparse, X_np)


def test_get_mask(failure_logger, mask_dataset):
    mask_value, X_np, X = mask_dataset
    cu_mask = cu_get_mask(X, value_to_mask=mask_value)
    sk_mask = sk_get_mask(X_np, value_to_mask=mask_value)
    assert_allclose(cu_mask, sk_mask)


def test_masked_column_median(failure_logger, mask_dataset):
    mask_value, X_np, X = mask_dataset
    median = _masked_column_median(X, mask_value).get()
    mask = ~sk_get_mask(X_np, value_to_mask=mask_value)
    n_columns = X.shape[1]
    for i in range(n_columns):
        column_mask = mask[:, i]
        column_median = np.median(X_np[:, i][column_mask])
        assert column_median == median[i]


def test_masked_column_mean(failure_logger, mask_dataset):
    mask_value, X_np, X = mask_dataset
    mean = _masked_column_mean(X, mask_value).get()
    mask = ~sk_get_mask(X_np, value_to_mask=mask_value)
    n_columns = X.shape[1]
    for i in range(n_columns):
        column_mask = mask[:, i]
        column_mean = np.mean(X_np[:, i][column_mask])
        assert column_mean == mean[i]


def test_masked_column_mode(failure_logger, mask_dataset):
    mask_value, X_np, X = mask_dataset
    mode = _masked_column_mode(X, mask_value).get()
    mask = ~sk_get_mask(X_np, value_to_mask=mask_value)
    n_columns = X.shape[1]
    for i in range(n_columns):
        column_mask = mask[:, i]
        column_mode = stats.mode(X_np[:, i][column_mask], keepdims=True)[0][0]
        assert column_mode == mode[i]

# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

from cuml.experimental.preprocessing import \
    StandardScaler as cuStandardScaler, \
    MinMaxScaler as cuMinMaxScaler, \
    MaxAbsScaler as cuMaxAbsScaler, \
    Normalizer as cuNormalizer, \
    Binarizer as cuBinarizer, \
    PolynomialFeatures as cuPolynomialFeatures, \
    SimpleImputer as cuSimpleImputer, \
    RobustScaler as cuRobustScaler, \
    KBinsDiscretizer as cuKBinsDiscretizer
from cuml.experimental.preprocessing import scale as cu_scale, \
                            minmax_scale as cu_minmax_scale, \
                            normalize as cu_normalize, \
                            add_dummy_feature as cu_add_dummy_feature, \
                            binarize as cu_binarize, \
                            robust_scale as cu_robust_scale
from sklearn.preprocessing import StandardScaler as skStandardScaler, \
                                  MinMaxScaler as skMinMaxScaler, \
                                  MaxAbsScaler as skMaxAbsScaler, \
                                  Normalizer as skNormalizer, \
                                  Binarizer as skBinarizer, \
                                  PolynomialFeatures as skPolynomialFeatures, \
                                  RobustScaler as skRobustScaler
from sklearn.preprocessing import scale as sk_scale, \
                                  minmax_scale as sk_minmax_scale, \
                                  normalize as sk_normalize, \
                                  add_dummy_feature as sk_add_dummy_feature, \
                                  binarize as sk_binarize, \
                                  robust_scale as sk_robust_scale
from sklearn.impute import SimpleImputer as skSimpleImputer
from sklearn.preprocessing import KBinsDiscretizer as skKBinsDiscretizer

from cuml.thirdparty_adapters.sparsefuncs_fast import \
    csr_mean_variance_axis0, \
    csc_mean_variance_axis0, \
    _csc_mean_variance_axis0, \
    inplace_csr_row_normalize_l1, \
    inplace_csr_row_normalize_l2

from cuml.test.test_preproc_utils import \
    clf_dataset, int_dataset, blobs_dataset, \
    sparse_clf_dataset, \
    sparse_blobs_dataset, \
    sparse_int_dataset  # noqa: F401
from cuml.test.test_preproc_utils import assert_allclose
from cuml.common.import_utils import check_cupy8

import numpy as np
import cupy as cp


def test_minmax_scaler(clf_dataset):  # noqa: F811
    X_np, X = clf_dataset

    scaler = cuMinMaxScaler(copy=True)
    t_X = scaler.fit_transform(X)
    r_X = scaler.inverse_transform(t_X)
    assert type(t_X) == type(X)
    assert type(r_X) == type(t_X)

    scaler = skMinMaxScaler(copy=True)
    sk_t_X = scaler.fit_transform(X_np)
    sk_r_X = scaler.inverse_transform(sk_t_X)

    assert_allclose(t_X, sk_t_X)
    assert_allclose(r_X, sk_r_X)


def test_minmax_scale(clf_dataset):  # noqa: F811
    X_np, X = clf_dataset

    t_X = cu_minmax_scale(X)
    assert type(t_X) == type(X)

    sk_t_X = sk_minmax_scale(X_np)

    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("with_mean", [True, False])
@pytest.mark.parametrize("with_std", [True, False])
def test_standard_scaler(clf_dataset, with_mean, with_std):  # noqa: F811
    X_np, X = clf_dataset

    scaler = cuStandardScaler(copy=True, with_mean=with_mean,
                              with_std=with_std)
    t_X = scaler.fit_transform(X)
    r_X = scaler.inverse_transform(t_X)
    assert type(t_X) == type(X)
    assert type(r_X) == type(t_X)

    scaler = skStandardScaler(copy=True, with_mean=with_mean,
                              with_std=with_std)
    sk_t_X = scaler.fit_transform(X_np)
    sk_r_X = scaler.inverse_transform(sk_t_X)

    assert_allclose(t_X, sk_t_X)
    assert_allclose(r_X, sk_r_X)


@pytest.mark.parametrize("with_std", [True, False])
def test_standard_scaler_sparse(sparse_clf_dataset, with_std):  # noqa: F811
    X_np, X = sparse_clf_dataset

    scaler = cuStandardScaler(copy=True, with_mean=False, with_std=with_std)
    t_X = scaler.fit_transform(X)
    r_X = scaler.inverse_transform(t_X)
    assert type(t_X) == type(X)
    assert type(r_X) == type(t_X)

    scaler = skStandardScaler(copy=True, with_mean=False, with_std=with_std)
    sk_t_X = scaler.fit_transform(X_np)
    sk_r_X = scaler.inverse_transform(sk_t_X)

    assert_allclose(t_X, sk_t_X)
    assert_allclose(r_X, sk_r_X)


@pytest.mark.parametrize("with_mean", [True, False])
@pytest.mark.parametrize("with_std", [True, False])
def test_scale(clf_dataset, with_mean, with_std):  # noqa: F811
    X_np, X = clf_dataset

    t_X = cu_scale(X, copy=True, with_mean=with_mean, with_std=with_std)
    assert type(t_X) == type(X)

    sk_t_X = sk_scale(X_np, copy=True, with_mean=with_mean, with_std=with_std)

    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("with_std", [True, False])
def test_scale_sparse(sparse_clf_dataset, with_std):  # noqa: F811
    X_np, X = sparse_clf_dataset

    t_X = cu_scale(X, copy=True, with_mean=False, with_std=with_std)
    assert type(t_X) == type(X)

    sk_t_X = sk_scale(X_np, copy=True, with_mean=False, with_std=with_std)

    assert_allclose(t_X, sk_t_X)


@check_cupy8('pytest')
def test_maxabs_scaler(clf_dataset):  # noqa: F811
    X_np, X = clf_dataset

    scaler = cuMaxAbsScaler(copy=True)
    t_X = scaler.fit_transform(X)
    r_X = scaler.inverse_transform(t_X)
    assert type(t_X) == type(X)
    assert type(r_X) == type(t_X)

    scaler = skMaxAbsScaler(copy=True)
    sk_t_X = scaler.fit_transform(X_np)
    sk_r_X = scaler.inverse_transform(sk_t_X)

    assert_allclose(t_X, sk_t_X)
    assert_allclose(r_X, sk_r_X)


@check_cupy8('pytest')
def test_maxabs_scaler_sparse(sparse_clf_dataset):  # noqa: F811
    X_np, X = sparse_clf_dataset

    scaler = cuMaxAbsScaler(copy=True)
    t_X = scaler.fit_transform(X)
    r_X = scaler.inverse_transform(t_X)
    assert type(t_X) == type(X)
    assert type(r_X) == type(t_X)

    scaler = skMaxAbsScaler(copy=True)
    sk_t_X = scaler.fit_transform(X_np)
    sk_r_X = scaler.inverse_transform(sk_t_X)

    assert_allclose(t_X, sk_t_X)
    assert_allclose(r_X, sk_r_X)


@check_cupy8('pytest')
@pytest.mark.parametrize("norm", ['l1', 'l2', 'max'])
def test_normalizer(clf_dataset, norm):  # noqa: F811
    X_np, X = clf_dataset

    normalizer = cuNormalizer(norm=norm, copy=True)
    t_X = normalizer.fit_transform(X)
    assert type(t_X) == type(X)

    normalizer = skNormalizer(norm=norm, copy=True)
    sk_t_X = normalizer.fit_transform(X_np)

    assert_allclose(t_X, sk_t_X)


@check_cupy8('pytest')
@pytest.mark.parametrize("norm", ['l1', 'l2', 'max'])
def test_normalizer_sparse(sparse_clf_dataset, norm):  # noqa: F811
    X_np, X = sparse_clf_dataset

    if X.format == 'csc':
        pytest.skip("Skipping CSC matrices")

    normalizer = cuNormalizer(norm=norm, copy=True)
    t_X = normalizer.fit_transform(X)
    assert type(t_X) == type(X)

    normalizer = skNormalizer(norm=norm, copy=True)
    sk_t_X = normalizer.fit_transform(X_np)

    assert_allclose(t_X, sk_t_X)


@check_cupy8('pytest')
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("norm", ['l1', 'l2', 'max'])
@pytest.mark.parametrize("return_norm", [True, False])
def test_normalize(clf_dataset, axis, norm, return_norm):  # noqa: F811
    X_np, X = clf_dataset

    if return_norm:
        t_X, t_norms = cu_normalize(X, axis=axis, norm=norm,
                                    return_norm=return_norm)
        sk_t_X, sk_t_norms = sk_normalize(X_np, axis=axis, norm=norm,
                                          return_norm=return_norm)
        assert_allclose(t_norms, sk_t_norms)
    else:
        t_X = cu_normalize(X, axis=axis, norm=norm, return_norm=return_norm)
        sk_t_X = sk_normalize(X_np, axis=axis, norm=norm,
                              return_norm=return_norm)

    assert type(t_X) == type(X)
    assert_allclose(t_X, sk_t_X)


@check_cupy8('pytest')
@pytest.mark.parametrize("norm", ['l1', 'l2', 'max'])
def test_normalize_sparse(sparse_clf_dataset, norm):  # noqa: F811
    X_np, X = sparse_clf_dataset

    axis = 0 if X.format == 'csc' else 1

    t_X = cu_normalize(X, axis=axis, norm=norm)
    assert type(t_X) == type(X)

    sk_t_X = sk_normalize(X_np, axis=axis, norm=norm)

    assert_allclose(t_X, sk_t_X)


@check_cupy8('pytest')
@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent",
                                      "constant"])
@pytest.mark.parametrize("missing_values", [0., 1., np.nan])
def test_imputer(int_dataset, strategy, missing_values):  # noqa: F811
    X_np, X = int_dataset
    fill_value = np.random.randint(10, size=1)[0]

    imputer = cuSimpleImputer(copy=True, missing_values=missing_values,
                              strategy=strategy, fill_value=fill_value)
    t_X = imputer.fit_transform(X)
    assert type(t_X) == type(X)

    imputer = skSimpleImputer(copy=True, missing_values=missing_values,
                              strategy=strategy, fill_value=fill_value)
    sk_t_X = imputer.fit_transform(X_np)

    assert_allclose(t_X, sk_t_X)


@check_cupy8('pytest')
@pytest.mark.parametrize("strategy", ["mean", "median", "most_frequent",
                         "constant"])
@pytest.mark.parametrize("missing_values", [np.nan, 1.])
def test_imputer_sparse(sparse_int_dataset, strategy,  # noqa: F811
                        missing_values):
    X_np, X = sparse_int_dataset

    if X.format == 'csr':
        pytest.skip("Skipping CSR matrices")

    X_sp = X_np.tocsc()

    if np.isnan(missing_values):
        # Adding nan when missing value is nan
        random_loc = np.random.choice(X.nnz,
                                      int(X.nnz * 0.1),
                                      replace=False)
        X_sp.data[random_loc] = np.nan
        X = X.copy()
        X.data[random_loc] = np.nan

    fill_value = np.random.randint(10, size=1)[0]

    imputer = cuSimpleImputer(copy=True, missing_values=missing_values,
                              strategy=strategy, fill_value=fill_value)
    t_X = imputer.fit_transform(X)
    assert type(t_X) == type(X)

    imputer = skSimpleImputer(copy=True, missing_values=missing_values,
                              strategy=strategy, fill_value=fill_value)
    sk_t_X = imputer.fit_transform(X_sp)
    assert_allclose(t_X, sk_t_X)


@check_cupy8('pytest')
@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("interaction_only", [True, False])
@pytest.mark.parametrize("include_bias", [True, False])
@pytest.mark.parametrize("order", ['C', 'F'])
def test_poly_features(clf_dataset, degree,  # noqa: F811
                       interaction_only, include_bias, order):
    X_np, X = clf_dataset

    polyfeatures = cuPolynomialFeatures(degree=degree, order=order,
                                        interaction_only=interaction_only,
                                        include_bias=include_bias)
    t_X = polyfeatures.fit_transform(X)
    assert type(X) == type(t_X)

    if isinstance(t_X, np.ndarray):
        if order == 'C':
            assert t_X.flags['C_CONTIGUOUS']
        elif order == 'F':
            assert t_X.flags['F_CONTIGUOUS']

    polyfeatures = skPolynomialFeatures(degree=degree, order=order,
                                        interaction_only=interaction_only,
                                        include_bias=include_bias)
    sk_t_X = polyfeatures.fit_transform(X_np)

    assert_allclose(t_X, sk_t_X, rtol=0.1, atol=0.1)


@check_cupy8('pytest')
@pytest.mark.parametrize("degree", [2, 3])
@pytest.mark.parametrize("interaction_only", [True, False])
@pytest.mark.parametrize("include_bias", [True, False])
def test_poly_features_sparse(sparse_clf_dataset, degree,  # noqa: F811
                              interaction_only, include_bias):
    X_np, X = sparse_clf_dataset

    polyfeatures = cuPolynomialFeatures(degree=degree,
                                        interaction_only=interaction_only,
                                        include_bias=include_bias)
    t_X = polyfeatures.fit_transform(X)
    assert type(t_X) == type(X)

    polyfeatures = skPolynomialFeatures(degree=degree,
                                        interaction_only=interaction_only,
                                        include_bias=include_bias)
    sk_t_X = polyfeatures.fit_transform(X_np)

    assert_allclose(t_X, sk_t_X, rtol=0.1, atol=0.1)


@pytest.mark.parametrize("value", [1.0, 42])
def test_add_dummy_feature(clf_dataset, value):  # noqa: F811
    X_np, X = clf_dataset

    t_X = cu_add_dummy_feature(X, value=value)
    assert type(t_X) == type(X)

    sk_t_X = sk_add_dummy_feature(X_np, value=value)
    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("value", [1.0, 42])
def test_add_dummy_feature_sparse(sparse_clf_dataset, value):  # noqa: F811
    X_np, X = sparse_clf_dataset

    t_X = cu_add_dummy_feature(X, value=value)
    assert type(t_X) == type(X)

    sk_t_X = sk_add_dummy_feature(X_np, value=value)
    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("threshold", [0., 1.])
def test_binarize(clf_dataset, threshold):  # noqa: F811
    X_np, X = clf_dataset

    t_X = cu_binarize(X, threshold=threshold, copy=True)
    assert type(t_X) == type(X)

    sk_t_X = sk_binarize(X_np, threshold=threshold, copy=True)

    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("threshold", [0., 1.])
def test_binarize_sparse(sparse_clf_dataset, threshold):  # noqa: F811
    X_np, X = sparse_clf_dataset

    t_X = cu_binarize(X, threshold=threshold, copy=True)
    assert type(t_X) == type(X)

    sk_t_X = sk_binarize(X_np, threshold=threshold, copy=True)

    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("threshold", [0., 1.])
def test_binarizer(clf_dataset, threshold):  # noqa: F811
    X_np, X = clf_dataset

    binarizer = cuBinarizer(threshold=threshold, copy=True)
    t_X = binarizer.fit_transform(X)
    assert type(t_X) == type(X)

    binarizer = skBinarizer(threshold=threshold, copy=True)
    sk_t_X = binarizer.fit_transform(X_np)

    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("threshold", [0., 1.])
def test_binarizer_sparse(sparse_clf_dataset, threshold):  # noqa: F811
    X_np, X = sparse_clf_dataset

    binarizer = cuBinarizer(threshold=threshold, copy=True)
    t_X = binarizer.fit_transform(X)
    assert type(t_X) == type(X)

    binarizer = skBinarizer(threshold=threshold, copy=True)
    sk_t_X = binarizer.fit_transform(X_np)

    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("with_centering", [True, False])
@pytest.mark.parametrize("with_scaling", [True, False])
@pytest.mark.parametrize("quantile_range", [(25., 75.), (10., 90.)])
def test_robust_scaler(clf_dataset, with_centering,  # noqa: F811
                       with_scaling, quantile_range):
    X_np, X = clf_dataset

    scaler = cuRobustScaler(with_centering=with_centering,
                            with_scaling=with_scaling,
                            quantile_range=quantile_range,
                            copy=True)
    t_X = scaler.fit_transform(X)
    r_X = scaler.inverse_transform(t_X)
    assert type(t_X) == type(X)
    assert type(r_X) == type(t_X)

    scaler = skRobustScaler(with_centering=with_centering,
                            with_scaling=with_scaling,
                            quantile_range=quantile_range,
                            copy=True)
    sk_t_X = scaler.fit_transform(X_np)
    sk_r_X = scaler.inverse_transform(sk_t_X)

    assert_allclose(t_X, sk_t_X)
    assert_allclose(r_X, sk_r_X)


@pytest.mark.parametrize("with_scaling", [True, False])
@pytest.mark.parametrize("quantile_range", [(25., 75.), (10., 90.)])
def test_robust_scaler_sparse(sparse_clf_dataset,  # noqa: F811
                              with_scaling, quantile_range):
    X_np, X = sparse_clf_dataset

    if X.format != 'csc':
        X = X.tocsc()

    scaler = cuRobustScaler(with_centering=False,
                            with_scaling=with_scaling,
                            quantile_range=quantile_range,
                            copy=True)
    t_X = scaler.fit_transform(X)
    r_X = scaler.inverse_transform(t_X)
    assert type(t_X) == type(X)
    assert type(r_X) == type(t_X)

    scaler = skRobustScaler(with_centering=False,
                            with_scaling=with_scaling,
                            quantile_range=quantile_range,
                            copy=True)
    sk_t_X = scaler.fit_transform(X_np)
    sk_r_X = scaler.inverse_transform(sk_t_X)

    assert_allclose(t_X, sk_t_X)
    assert_allclose(r_X, sk_r_X)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("with_centering", [True, False])
@pytest.mark.parametrize("with_scaling", [True, False])
@pytest.mark.parametrize("quantile_range", [(25., 75.), (10., 90.)])
def test_robust_scale(clf_dataset, with_centering,  # noqa: F811
                      axis, with_scaling, quantile_range):
    X_np, X = clf_dataset

    t_X = cu_robust_scale(X, axis=axis,
                          with_centering=with_centering,
                          with_scaling=with_scaling,
                          quantile_range=quantile_range,
                          copy=True)
    assert type(t_X) == type(X)

    sk_t_X = sk_robust_scale(X_np, axis=axis,
                             with_centering=with_centering,
                             with_scaling=with_scaling,
                             quantile_range=quantile_range,
                             copy=True)

    assert_allclose(t_X, sk_t_X)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("with_scaling", [True, False])
@pytest.mark.parametrize("quantile_range", [(25., 75.), (10., 90.)])
def test_robust_scale_sparse(sparse_clf_dataset,  # noqa: F811
                             axis, with_scaling, quantile_range):
    X_np, X = sparse_clf_dataset

    if X.format != 'csc' and axis == 0:
        X = X.tocsc()
    elif X.format != 'csr' and axis == 1:
        X = X.tocsr()

    t_X = cu_robust_scale(X, axis=axis,
                          with_centering=False,
                          with_scaling=with_scaling,
                          quantile_range=quantile_range,
                          copy=True)
    assert type(t_X) == type(X)

    sk_t_X = sk_robust_scale(X_np, axis=axis,
                             with_centering=False,
                             with_scaling=with_scaling,
                             quantile_range=quantile_range,
                             copy=True)

    assert_allclose(t_X, sk_t_X)


@check_cupy8('pytest')
@pytest.mark.parametrize("n_bins", [5, 20])
@pytest.mark.parametrize("encode", ['ordinal', 'onehot-dense', 'onehot'])
@pytest.mark.parametrize("strategy", [
    pytest.param('uniform', marks=pytest.mark.xfail(
        strict=False,
        reason='Intermittent mismatch with sklearn'
        ' (https://github.com/rapidsai/cuml/issues/3481)'
    )),
    pytest.param('quantile', marks=pytest.mark.xfail(
        strict=False,
        reason='Bug in cupy.percentile'
        ' (https://github.com/cupy/cupy/issues/4607)'
    )),
    'kmeans'
])
def test_kbinsdiscretizer(blobs_dataset, n_bins,  # noqa: F811
                          encode, strategy):
    X_np, X = blobs_dataset

    transformer = cuKBinsDiscretizer(n_bins=n_bins,
                                     encode=encode,
                                     strategy=strategy)
    t_X = transformer.fit_transform(X)
    r_X = transformer.inverse_transform(t_X)

    if encode != 'onehot':
        assert type(t_X) == type(X)
        assert type(r_X) == type(t_X)

    transformer = skKBinsDiscretizer(n_bins=n_bins,
                                     encode=encode,
                                     strategy=strategy)
    sk_t_X = transformer.fit_transform(X_np)
    sk_r_X = transformer.inverse_transform(sk_t_X)

    if strategy == 'kmeans':
        assert_allclose(t_X, sk_t_X, ratio_tol=0.2)
    else:
        assert_allclose(t_X, sk_t_X)
        assert_allclose(r_X, sk_r_X)


def test_csr_mean_variance_axis0(sparse_clf_dataset):  # noqa: F811
    X_np, X = sparse_clf_dataset

    if not cp.sparse.issparse(X):
        pytest.skip("Skipping non-CuPy or non-sparse arrays")

    if X.format != 'csr':
        X = X.tocsr()

    means, variances = csr_mean_variance_axis0(X)

    X_np = X_np.toarray()
    ref_means = np.nanmean(X_np, axis=0)
    ref_variances = np.nanvar(X_np, axis=0)

    assert_allclose(means, ref_means)
    assert_allclose(variances, ref_variances)


def test_csc_mean_variance_axis0(sparse_clf_dataset):  # noqa: F811
    X_np, X = sparse_clf_dataset

    if not cp.sparse.issparse(X):
        pytest.skip("Skipping non-CuPy or non-sparse arrays")

    if X.format != 'csc':
        X = X.tocsc()

    means, variances = csc_mean_variance_axis0(X)

    X_np = X_np.toarray()
    ref_means = np.nanmean(X_np, axis=0)
    ref_variances = np.nanvar(X_np, axis=0)

    assert_allclose(means, ref_means)
    assert_allclose(variances, ref_variances)


def test__csc_mean_variance_axis0(sparse_clf_dataset):  # noqa: F811
    X_np, X = sparse_clf_dataset

    if not cp.sparse.issparse(X):
        pytest.skip("Skipping non-CuPy or non-sparse arrays")

    if X.format != 'csc':
        X = X.tocsc()

    means, variances, counts_nan = _csc_mean_variance_axis0(X)

    X_np = X_np.toarray()
    ref_means = np.nanmean(X_np, axis=0)
    ref_variances = np.nanvar(X_np, axis=0)
    ref_counts_nan = np.isnan(X_np).sum(axis=0)

    assert_allclose(means, ref_means)
    assert_allclose(variances, ref_variances)
    assert_allclose(counts_nan, ref_counts_nan)


def test_inplace_csr_row_normalize_l1(sparse_clf_dataset):  # noqa: F811
    X_np, X = sparse_clf_dataset

    if not cp.sparse.issparse(X):
        pytest.skip("Skipping non-CuPy or non-sparse arrays")

    if X.format != 'csr':
        X = X.tocsr()

    inplace_csr_row_normalize_l1(X)

    X_np = X_np.toarray()
    X_np = sk_normalize(X_np, norm='l1', axis=1)

    assert_allclose(X, X_np)


def test_inplace_csr_row_normalize_l2(sparse_clf_dataset):  # noqa: F811
    X_np, X = sparse_clf_dataset

    if not cp.sparse.issparse(X):
        pytest.skip("Skipping non-CuPy or non-sparse arrays")

    if X.format != 'csr':
        X = X.tocsr()

    inplace_csr_row_normalize_l2(X)

    X_np = X_np.toarray()
    X_np = sk_normalize(X_np, norm='l2', axis=1)

    assert_allclose(X, X_np)


def test__repr__():
    assert cuStandardScaler().__repr__() == 'StandardScaler()'
    assert cuMinMaxScaler().__repr__() == 'MinMaxScaler()'
    assert cuMaxAbsScaler().__repr__() == 'MaxAbsScaler()'
    assert cuNormalizer().__repr__() == 'Normalizer()'
    assert cuBinarizer().__repr__() == 'Binarizer()'
    assert cuPolynomialFeatures().__repr__() == 'PolynomialFeatures()'
    assert cuSimpleImputer().__repr__() == 'SimpleImputer()'
    assert cuRobustScaler().__repr__() == 'RobustScaler()'
    assert cuKBinsDiscretizer().__repr__() == 'KBinsDiscretizer()'

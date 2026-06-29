#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import math
from itertools import combinations

import cupy as cp
import numpy as np
import pytest
import scipy.special
import sklearn.neighbors
import sklearn.svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

import cuml
from cuml import KernelExplainer, Lasso
from cuml.testing.datasets import with_dtype
from cuml.testing.utils import ClassEnumerator, get_shap_values

models_config = ClassEnumerator(module=cuml)
models = models_config.get_models()

_REFERENCE_POWERSET_MAX_FEATURES = 6
_REFERENCE_POWERSET_MAX_ROWS = 2**_REFERENCE_POWERSET_MAX_FEATURES - 2


def _as_numpy(value):
    if isinstance(value, cp.ndarray):
        return cp.asnumpy(value)
    if hasattr(value, "to_numpy"):
        return value.to_numpy()
    if hasattr(value, "get"):
        return value.get()
    return np.asarray(value)


def _assert_additive_shap_values(
    shap_values,
    prediction,
    expected_value,
    *,
    expected_shape,
    rtol=1e-5,
    atol=1e-5,
):
    shap_values = _as_numpy(shap_values)
    prediction = np.asarray(_as_numpy(prediction)).squeeze()
    expected_value = np.asarray(_as_numpy(expected_value)).squeeze()

    assert shap_values.shape == expected_shape
    assert np.all(np.isfinite(shap_values))
    np.testing.assert_allclose(
        np.sum(shap_values, axis=-1),
        prediction - expected_value,
        rtol=rtol,
        atol=atol,
    )


def _reference_powerset(n, r, nrows, full_powerset=False, dtype=np.float32):
    """Build the expected Kernel SHAP mask matrix and mask weights.

    Each row in the mask matrix marks which features are present for one
    synthetic model evaluation: 1 means the feature is present, 0 means it is
    hidden. This mirrors the subset ordering used by `_powerset`, but derives
    the expected rows from `itertools.combinations` instead of storing a
    hard-coded result table.

    When `full_powerset` is false, each generated subset is followed by its
    complement, matching the sampled/exact-pair layout used by Kernel SHAP.
    """
    if (
        n > _REFERENCE_POWERSET_MAX_FEATURES
        or nrows > _REFERENCE_POWERSET_MAX_ROWS
    ):
        raise ValueError(
            "_reference_powerset is only intended for trivial test cases "
            f"(n <= {_REFERENCE_POWERSET_MAX_FEATURES} and "
            f"nrows <= {_REFERENCE_POWERSET_MAX_ROWS}); got n={n}, "
            f"nrows={nrows}."
        )

    result = np.zeros((nrows, n), dtype=dtype)
    weights = np.zeros(nrows, dtype=dtype)
    idx = 0
    upper_limit = n if full_powerset else r
    for subset_size in range(1, upper_limit):
        for subset in combinations(range(n), subset_size):
            result[idx, subset] = 1
            weights[idx] = _reference_shapley_kernel(n, subset_size)
            if not full_powerset:
                result[idx + 1] = 1 - result[idx]
                weights[idx + 1] = _reference_shapley_kernel(n, subset_size)
                idx += 1
            idx += 1
    return result, weights


def _reference_shapley_kernel(n_features, subset_size):
    """Compute the expected Kernel SHAP weight for one subset size.

    Kernel SHAP assigns each feature mask a weight that depends only on the
    total feature count and how many features are present in the mask. The
    empty and full masks use the same large finite sentinel as cuML's
    implementation to avoid division by zero at the formula boundaries.
    """
    if subset_size == 0 or subset_size == n_features:
        return 10000
    return (n_features - 1) / (
        scipy.special.binom(n_features, subset_size)
        * subset_size
        * (n_features - subset_size)
    )


###############################################################################
#                              End to end tests                               #
###############################################################################


@pytest.mark.parametrize(
    "model", [cuml.LinearRegression, cuml.KNeighborsRegressor, cuml.SVR]
)
def test_exact_regression_datasets(exact_shap_regression_dataset, model):
    X_train, X_test, y_train, y_test = exact_shap_regression_dataset

    models = []
    models.append(model().fit(X_train, y_train))
    models.append(cuml_skl_class_dict[model]().fit(X_train, y_train))

    for mod in models:
        explainer, shap_values = get_shap_values(
            model=mod.predict,
            background_dataset=X_train,
            explained_dataset=X_test,
            explainer=KernelExplainer,
        )
        fx = mod.predict(X_test)
        for i in range(3):
            _assert_additive_shap_values(
                shap_values[i],
                fx[i],
                explainer.expected_value,
                expected_shape=(X_test.shape[1],),
                rtol=1e-4,
                atol=1e-4,
            )


@pytest.mark.parametrize("cls", [cuml.SVC, sklearn.svm.SVC])
def test_exact_classification_datasets(exact_shap_classification_dataset, cls):
    X_train, X_test, y_train, y_test = exact_shap_classification_dataset

    model = CalibratedClassifierCV(cls(), ensemble=False)
    model.fit(X_train, y_train)

    explainer, shap_values = get_shap_values(
        model=model.predict_proba,
        background_dataset=X_train,
        explained_dataset=X_test,
        explainer=KernelExplainer,
    )

    fx = model.predict_proba(X_test)
    for idx, svs in enumerate(shap_values):
        _assert_additive_shap_values(
            svs,
            fx[:, idx],
            explainer.expected_value[idx],
            expected_shape=(X_test.shape[0], X_test.shape[1]),
            rtol=1e-4,
            atol=1e-4,
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_features", [10, 30])
@pytest.mark.parametrize("n_background", [10, 30])
@pytest.mark.parametrize("model", [cuml.TruncatedSVD, cuml.PCA])
def test_kernel_shap_standalone(dtype, n_features, n_background, model):
    X, y = with_dtype(
        make_regression(
            n_samples=n_background + 3,
            n_features=n_features,
            noise=0.1,
            random_state=42,
        ),
        dtype,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=3, random_state=42
    )

    mod = model(n_components=3).fit(X_train, y_train)
    explainer, shap_values = get_shap_values(
        model=mod.transform,
        background_dataset=X_train,
        explained_dataset=X_test,
        explainer=KernelExplainer,
    )

    exp_v = explainer.expected_value

    # transform returns one SHAP value array per component.
    fx = mod.transform(X_test)
    for comp_idx in range(3):
        _assert_additive_shap_values(
            shap_values[comp_idx],
            _as_numpy(fx)[:, comp_idx],
            exp_v[comp_idx],
            expected_shape=(X_test.shape[0], n_features),
            rtol=1e-5,
            atol=1e-5,
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_features", [11, 15])
@pytest.mark.parametrize("n_background", [30])
@pytest.mark.parametrize("model", [cuml.SVR])
def test_kernel_gpu_cpu_shap(dtype, n_features, n_background, model):
    shap = pytest.importorskip("shap")

    X, y = with_dtype(
        make_regression(
            n_samples=n_background + 3,
            n_features=n_features,
            noise=0.1,
            random_state=42,
        ),
        dtype,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=3, random_state=42
    )

    mod = model().fit(X_train, y_train)
    explainer, shap_values = get_shap_values(
        model=mod.predict,
        background_dataset=X_train,
        explained_dataset=X_test,
        explainer=KernelExplainer,
    )

    exp_v = explainer.expected_value

    fx = mod.predict(X_test)
    _assert_additive_shap_values(
        shap_values,
        fx,
        exp_v,
        expected_shape=(X_test.shape[0], X_test.shape[1]),
        rtol=1e-5,
        atol=1e-5,
    )

    explainer = shap.KernelExplainer(mod.predict, cp.asnumpy(X_train))
    cpu_shap_values = explainer.shap_values(cp.asnumpy(X_test))

    np.testing.assert_allclose(
        _as_numpy(shap_values), cpu_shap_values, rtol=1e-1, atol=1e-1
    )


def test_kernel_regression_dataset():
    # Generate synthetic regression dataset (similar to California housing)
    X, y = make_regression(
        n_samples=20640, n_features=8, noise=0.5, random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # making all float32 to use gpu predict on random forest
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    cumodel = cuml.RandomForestRegressor(max_features="sqrt").fit(
        X_train, y_train
    )

    explainer = KernelExplainer(
        model=cumodel.predict, data=X_train[:100], output_type="numpy"
    )

    cu_shap_values = explainer.shap_values(X_test[:2])

    _assert_additive_shap_values(
        cu_shap_values,
        cumodel.predict(X_test[:2]),
        explainer.expected_value,
        expected_shape=(2, X_test.shape[1]),
        rtol=1e-4,
        atol=1e-4,
    )


###############################################################################
#                        Single function unit tests                           #
###############################################################################


def test_binom_coef():
    for i in range(1, 101):
        val = cuml.explainer.kernel_shap._binomCoef(100, i)
        assert math.isclose(val, scipy.special.binom(100, i), rel_tol=1e-15)


def test_shapley_kernel():
    for i in range(11):
        val = cuml.explainer.kernel_shap._shapley_kernel(10, i)
        assert math.isclose(val, _reference_shapley_kernel(10, i))


def test_full_powerset():
    ps, w = cuml.explainer.kernel_shap._powerset(
        5, 2, 2**5 - 2, full_powerset=True
    )
    expected_ps, expected_w = _reference_powerset(
        5, 2, 2**5 - 2, full_powerset=True
    )

    np.testing.assert_array_equal(ps, expected_ps)
    np.testing.assert_allclose(w, expected_w)


def test_partial_powerset():
    ps, w = cuml.explainer.kernel_shap._powerset(6, 3, 42)
    expected_ps, expected_w = _reference_powerset(6, 3, 42)

    np.testing.assert_array_equal(ps, expected_ps)
    np.testing.assert_allclose(w, expected_w)


def test_reference_powerset_rejects_nontrivial_size():
    with pytest.raises(ValueError, match="only intended for trivial"):
        _reference_powerset(7, 2, 2**7 - 2, full_powerset=True)


@pytest.mark.parametrize("full_powerset", [True, False])
def test_get_number_of_exact_random_samples(full_powerset):
    if full_powerset:
        (
            nsamples_exact,
            nsamples_random,
            ind,
        ) = cuml.explainer.kernel_shap._get_number_of_exact_random_samples(
            10, 2**10 + 1
        )
        assert nsamples_exact == 1022
        assert nsamples_random == 0
        assert ind == 5
    else:
        (
            nsamples_exact,
            nsamples_random,
            ind,
        ) = cuml.explainer.kernel_shap._get_number_of_exact_random_samples(
            10, 100
        )

        assert nsamples_exact == 20
        assert nsamples_random == 80
        assert ind == 2


def test_generate_nsamples_weights():
    samples, w = cuml.explainer.kernel_shap._generate_nsamples_weights(
        ncols=20,
        nsamples=30,
        nsamples_exact=10,
        nsamples_random=20,
        randind=5,
        dtype=np.float32,
    )
    # check that all our samples are between 5 and 6, and the weights in pairs
    # are generated correctly
    for i, s in enumerate(samples):
        assert s in [5, 6]
        assert np.isclose(
            float(w[i * 2]),
            cuml.explainer.kernel_shap._shapley_kernel(20, int(s)),
        )
        assert np.isclose(
            float(w[i * 2 + 1]),
            cuml.explainer.kernel_shap._shapley_kernel(20, int(s)),
        )


@pytest.mark.parametrize(
    "l1_type", ["auto", "aic", "bic", "num_features(3)", 0.2]
)
def test_l1_regularization(exact_shap_regression_dataset, l1_type):
    # currently this is a code test, not mathematical results test.
    # Hard to test without falling into testing the underlying algorithms
    # which are out of this unit test scope.
    X, w = cuml.explainer.kernel_shap._powerset(
        5, 2, 2**5 - 2, full_powerset=True
    )

    y = cp.random.rand(X.shape[0])
    nz = cuml.explainer.kernel_shap._l1_regularization(
        X=cp.asarray(X).astype(np.float32),
        y=cp.asarray(y).astype(np.float32),
        weights=cp.asarray(w),
        expected_value=0.0,
        fx=0.0,
        link_fn=cuml.explainer.common.identity,
        l1_reg=l1_type,
    )
    assert isinstance(nz, cp.ndarray)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.filterwarnings("ignore:Changing solver.*:UserWarning")
@pytest.mark.filterwarnings(
    "ignore:overflow encountered in divide:RuntimeWarning"
)
def test_typeerror_input():
    X, y = make_regression(n_samples=100, n_features=10, random_state=10)
    clf = Lasso()
    clf.fit(X, y)
    exp = KernelExplainer(model=clf.predict, data=X, nsamples=10)
    try:
        exp.shap_values(X)
    except ValueError as error:
        error_str = str(error)
        if (
            "operands could not be broadcast together" in error_str
            or "dimension must be fixed to" in error_str
        ):
            pytest.xfail(
                "Known sklearn LARS broadcasting bug - see scikit-learn#9603"
            )
        else:
            raise error


cuml_skl_class_dict = {
    cuml.LinearRegression: sklearn.linear_model.LinearRegression,
    cuml.KNeighborsRegressor: sklearn.neighbors.KNeighborsRegressor,
    cuml.SVR: sklearn.svm.SVR,
}

#
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#


import cupy as cp
import numpy as np
import pytest
import sklearn.neighbors
import sklearn.svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

import cuml
from cuml import PermutationExplainer
from cuml.testing.datasets import with_dtype
from cuml.testing.utils import ClassEnumerator, get_shap_values

models_config = ClassEnumerator(module=cuml)
models = models_config.get_models()


def _as_numpy(value):
    if isinstance(value, cp.ndarray):
        return cp.asnumpy(value)
    if hasattr(value, "to_numpy"):
        return value.to_numpy()
    if hasattr(value, "get"):
        return value.get()
    return np.asarray(value)


def _as_1d_numpy(value):
    return np.asarray(_as_numpy(value)).squeeze()


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


###############################################################################
#                              End to end tests                               #
###############################################################################


@pytest.mark.parametrize(
    "model", [cuml.LinearRegression, cuml.KNeighborsRegressor, cuml.SVR]
)
def test_regression_datasets(exact_shap_regression_dataset, model):
    X_train, X_test, y_train, y_test = exact_shap_regression_dataset

    models = []
    models.append(model().fit(X_train, y_train))
    models.append(cuml_skl_class_dict[model]().fit(X_train, y_train))

    for mod in models:
        explainer, shap_values = get_shap_values(
            model=mod.predict,
            background_dataset=X_train,
            explained_dataset=X_test,
            explainer=PermutationExplainer,
        )

        fx = mod.predict(X_test)
        exp_v = explainer.expected_value

        _assert_additive_shap_values(
            shap_values,
            fx,
            exp_v,
            expected_shape=(X_test.shape[0], X_test.shape[1]),
            rtol=1e-5,
            atol=1e-5,
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
        explainer=PermutationExplainer,
    )

    fx = model.predict_proba(X_test)
    exp_v = explainer.expected_value

    for class_idx, class_shap_values in enumerate(shap_values):
        _assert_additive_shap_values(
            class_shap_values,
            fx[:, class_idx],
            exp_v[class_idx],
            expected_shape=(X_test.shape[0], X_test.shape[1]),
            rtol=1e-5,
            atol=1e-5,
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_features", [11, 50])
@pytest.mark.parametrize("n_background", [10, 50])
@pytest.mark.parametrize("model", [cuml.LinearRegression, cuml.SVR])
@pytest.mark.parametrize("npermutations", [20])
def test_different_parameters(
    dtype, n_features, n_background, model, npermutations
):
    X, y = with_dtype(
        make_regression(
            n_samples=n_background + 5,
            n_features=n_features,
            noise=0.1,
            random_state=42,
        ),
        dtype,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=5, random_state=42
    )

    mod = model().fit(X_train, y_train)

    cu_explainer = PermutationExplainer(
        model=mod.predict, data=X_train, is_gpu_model=True
    )

    cu_shap_values = cu_explainer.shap_values(
        X_test, npermutations=npermutations
    )

    exp_v = float(cu_explainer.expected_value)
    fx = mod.predict(X_test)
    _assert_additive_shap_values(
        cu_shap_values,
        fx,
        exp_v,
        expected_shape=(X_test.shape[0], X_test.shape[1]),
        rtol=1e-3,
        atol=1e-3,
    )


###############################################################################
#                              Functional tests                               #
###############################################################################


def test_not_shuffled_explanation(exact_shap_regression_dataset):
    # in general permutation shap does not behave as predictable as
    # kernel shap, even when comparing permutation against kernel SHAP of the
    # mainline SHAP package. So these tests assure us that we're doing the
    # correct calculations, even if we can't compare directly.
    X_train, X_test, y_train, y_test = exact_shap_regression_dataset

    mod = cuml.LinearRegression().fit(X_train, y_train)

    explainer = PermutationExplainer(model=mod.predict, data=X_train)

    shap_values = explainer.shap_values(
        X_test[0], npermutations=1, testing=True
    )
    expected_shap_values = _as_numpy(mod.coef_) * (
        X_test[0] - np.mean(X_train, axis=0)
    )

    np.testing.assert_allclose(
        _as_1d_numpy(shap_values),
        _as_1d_numpy(expected_shap_values),
        rtol=1e-4,
        atol=1e-4,
    )


# Test against exact shap values for linear regression
# 1 permutation should give exact result
def test_permutation(exact_shap_regression_dataset):
    X_train, _, y_train, _ = exact_shap_regression_dataset
    # Train arbitrary model to get some coefficients
    mod = cuml.LinearRegression().fit(X_train, y_train)
    # Single background and foreground instance
    # Gives zero effect to features when they are 'off'
    # and the effect of the regression coefficient when they are 'on'
    X_background = np.zeros((1, X_train.shape[1]))
    X_foreground = np.ones((1, X_train.shape[1]))
    explainer = PermutationExplainer(model=mod.predict, data=X_background)

    shap_values = explainer.shap_values(
        X_foreground,
        npermutations=5,
    )

    np.testing.assert_allclose(
        _as_1d_numpy(mod.coef_),
        _as_1d_numpy(shap_values),
        rtol=1e-4,
        atol=1e-4,
    )


###############################################################################
#                                 Precomputed results                         #
#                               and testing variables                         #
###############################################################################

cuml_skl_class_dict = {
    cuml.LinearRegression: sklearn.linear_model.LinearRegression,
    cuml.KNeighborsRegressor: sklearn.neighbors.KNeighborsRegressor,
    cuml.SVR: sklearn.svm.SVR,
}

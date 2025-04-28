#
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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


import cupy as cp
import numpy as np
import pytest
import sklearn.neighbors
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

import cuml
from cuml import PermutationExplainer
from cuml.testing.datasets import with_dtype
from cuml.testing.utils import ClassEnumerator, get_shap_values

models_config = ClassEnumerator(module=cuml)
models = models_config.get_models()


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

        for i in range(3):
            assert (
                np.sum(cp.asnumpy(shap_values[i])) - abs(fx[i] - exp_v)
            ) <= 1e-5


def test_exact_classification_datasets(exact_shap_classification_dataset):
    X_train, X_test, y_train, y_test = exact_shap_classification_dataset

    models = []
    models.append(cuml.SVC(probability=True).fit(X_train, y_train))
    models.append(sklearn.svm.SVC(probability=True).fit(X_train, y_train))

    for mod in models:
        explainer, shap_values = get_shap_values(
            model=mod.predict_proba,
            background_dataset=X_train,
            explained_dataset=X_test,
            explainer=PermutationExplainer,
        )

        fx = mod.predict_proba(X_test)
        exp_v = explainer.expected_value

        for i in range(3):
            print(i, fx[i][1], shap_values[1][i])
            assert (
                np.sum(cp.asnumpy(shap_values[0][i]))
                - abs(fx[i][0] - exp_v[0])
            ) <= 1e-5
            assert (
                np.sum(cp.asnumpy(shap_values[1][i]))
                - abs(fx[i][1] - exp_v[1])
            ) <= 1e-5


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
    for i in range(5):
        assert (
            0.97
            <= (
                abs(np.sum(cp.asnumpy(cu_shap_values[i]))) / abs(fx[i] - exp_v)
            )
            <= 1.03
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

    assert np.allclose(
        shap_values, not_shuffled_shap_values, rtol=1e-04, atol=1e-04
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

    assert np.allclose(mod.coef_, shap_values, rtol=1e-04, atol=1e-04)


###############################################################################
#                                 Precomputed results                         #
#                               and testing variables                         #
###############################################################################

cuml_skl_class_dict = {
    cuml.LinearRegression: sklearn.linear_model.LinearRegression,
    cuml.KNeighborsRegressor: sklearn.neighbors.KNeighborsRegressor,
    cuml.SVR: sklearn.svm.SVR,
}

# values were precomputed with python code and with a modified version
# of SHAP's permutationExplainer that did not shuffle the indexes for the
# permutations, giving us a test of the calculations in our implementation
not_shuffled_shap_values = [
    -1.3628101e00,
    -1.0234560e02,
    1.3428497e-01,
    -6.1764000e01,
    2.6702881e-04,
    -3.4455948e00,
    -1.0159061e02,
    3.4058895e00,
    4.1598404e01,
    7.2152489e01,
    -2.1964169e00,
]

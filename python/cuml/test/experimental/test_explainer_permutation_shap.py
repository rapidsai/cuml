#
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


import cuml
import cuml.experimental.explainer
import cupy as cp
import numpy as np
import pytest
import sklearn.neighbors

from cuml.test.utils import ClassEnumerator
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

models_config = ClassEnumerator(module=cuml)
models = models_config.get_models()


@pytest.fixture(scope="module")
def exact_tests_dataset():
    X, y = make_regression(n_samples=101,
                           n_features=11,
                           noise=0.1,
                           random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1, random_state=42)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)
    return X_train, X_test, y_train, y_test


###############################################################################
#                              End to end tests                               #
###############################################################################


@pytest.mark.parametrize("model", [cuml.LinearRegression,
                                   cuml.KNeighborsRegressor,
                                   cuml.SVR])
def test_regression_datasets(exact_tests_dataset, model):
    # in general permutation shap does not behave as predictable as
    # kernel shap, even when comparing permutation against kernel SHAP of the
    # mainline SHAP package. So these tests assure us that we're doing the
    # correct calculations, even if we can't compare directly.
    X_train, X_test, y_train, y_test = exact_tests_dataset

    mod = model().fit(X_train, y_train)

    explainer = cuml.experimental.explainer.PermutationExplainer(
        model=mod.predict,
        data=X_train)

    cu_shap_values = explainer.shap_values(X_test)

    exp_v = float(explainer.expected_value)
    fx = mod.predict(X_test)
    assert (np.sum(cp.asnumpy(cu_shap_values)) - abs(fx - exp_v)) <= 1e-5

    skmod = cuml_skl_class_dict[model]().fit(X_train, y_train)

    explainer = cuml.experimental.explainer.PermutationExplainer(
        model=skmod.predict,
        data=X_train)

    skl_shap_values = explainer.shap_values(X_test)
    exp_v = float(explainer.expected_value)
    fx = mod.predict(X_test)
    assert (np.sum(cp.asnumpy(skl_shap_values)) - abs(fx - exp_v)) <= 1e-5


def test_exact_classification_datasets():
    X, y = make_classification(n_samples=101,
                               n_features=11,
                               random_state=42,
                               n_informative=2,
                               n_classes=2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1, random_state=42)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    mod = cuml.SVC(probability=True).fit(X_train, y_train)

    explainer = cuml.experimental.explainer.PermutationExplainer(
        model=mod.predict_proba,
        data=X_train)

    cu_shap_values = explainer.shap_values(X_test)

    exp_v = explainer.expected_value
    fx = mod.predict_proba(X_test)[0]
    assert (np.sum(cp.asnumpy(
        cu_shap_values[0])) - abs(fx[0] - exp_v[0])) <= 1e-5
    assert (np.sum(cp.asnumpy(
        cu_shap_values[1])) - abs(fx[1] - exp_v[1])) <= 1e-5

    mod = sklearn.svm.SVC(probability=True).fit(X_train, y_train)

    explainer = cuml.experimental.explainer.PermutationExplainer(
        model=mod.predict_proba,
        data=X_train)

    skl_shap_values = explainer.shap_values(X_test)

    exp_v = explainer.expected_value
    fx = mod.predict_proba(X_test)[0]
    assert (np.sum(cp.asnumpy(
        skl_shap_values[0])) - abs(fx[0] - exp_v[0])) <= 1e-5
    assert (np.sum(cp.asnumpy(
        skl_shap_values[1])) - abs(fx[1] - exp_v[1])) <= 1e-5


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("nfeatures", [11, 50])
@pytest.mark.parametrize("nbackground", [10, 50])
@pytest.mark.parametrize("model", [cuml.LinearRegression,
                                   cuml.SVR])
@pytest.mark.parametrize("npermutations", [5, 50])
def test_different_parameters(dtype, nfeatures, nbackground, model,
                              npermutations):
    X, y = cuml.datasets.make_regression(n_samples=nbackground + 5,
                                         n_features=nfeatures,
                                         noise=0.1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=5, random_state=42)

    X_train = X_train.astype(dtype)
    X_test = X_test.astype(dtype)
    y_train = y_train.astype(dtype)
    y_test = y_test.astype(dtype)

    mod = model().fit(X_train, y_train)

    cu_explainer = \
        cuml.experimental.explainer.PermutationExplainer(model=mod.predict,
                                                         data=X_train,
                                                         is_gpu_model=True)

    cu_shap_values = cu_explainer.shap_values(X_test,
                                              npermutations=npermutations)

    exp_v = float(cu_explainer.expected_value)
    fx = mod.predict(X_test)
    for i in range(5):
        assert 0.99 <= (abs(np.sum(cp.asnumpy(
            cu_shap_values[i]))) / abs(fx[i] - exp_v)) <= 1.01


###############################################################################
#                              Functional tests                               #
###############################################################################

def test_not_shuffled_explanation(exact_tests_dataset):
    # in general permutation shap does not behave as predictable as
    # kernel shap, even when comparing permutation against kernel SHAP of the
    # mainline SHAP package. So these tests assure us that we're doing the
    # correct calculations, even if we can't compare directly.
    X_train, X_test, y_train, y_test = exact_tests_dataset

    mod = cuml.LinearRegression().fit(X_train, y_train)

    explainer = cuml.experimental.explainer.PermutationExplainer(
        model=mod.predict,
        data=X_train)

    shap_values = explainer.shap_values(
        X_test,
        npermutations=1,
        testing=True
    )

    assert np.allclose(shap_values, not_shuffled_shap_values,
                       rtol=1e-04, atol=1e-04)


# Test against exact shap values for linear regression
# 1 permutation should give exact result
def test_permutation(exact_tests_dataset):
    X_train, X_test, y_train, y_test = exact_tests_dataset
    # Train arbitrary model to get some coefficients
    mod = cuml.LinearRegression().fit(X_train, y_train)
    # Single background and foreground instance
    # Gives zero effect to features when they are 'off'
    # and the effect of the regression coefficient when they are 'on'
    X_background = np.zeros((1, X_train.shape[1]))
    X_foreground = np.ones((1, X_train.shape[1]))
    explainer = cuml.experimental.explainer.PermutationExplainer(
        model=mod.predict,
        data=X_background)

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
    cuml.SVR: sklearn.svm.SVR
}

# values were precomputed with python code and with a modified version
# of SHAP's permutationExplainer that did not shuffle the indexes for the
# permutations, giving us a test of the calculations in our implementation
not_shuffled_shap_values = [
    -3.60017776e-01, -1.02140656e+02, 1.29915714e+00, -6.30791473e+01,
    2.47955322e-04, -2.31356430e+00, -1.01764305e+02, 3.39929199e+00,
    4.10347061e+01, 7.13340759e+01, -1.60478973e+00
]

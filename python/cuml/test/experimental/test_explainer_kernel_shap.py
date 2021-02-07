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
import math
import pytest
import sklearn.neighbors

from cuml.common.import_utils import has_scipy
from cuml.common.import_utils import has_shap
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


def experimental_test_and_log(cu_shap_values,
                              golden_result_values,
                              fx,
                              expected,
                              tolerance=1e-02):
    close_values = \
        np.allclose(cu_shap_values, golden_result_values,
                    rtol=tolerance, atol=tolerance)

    expected_sum = np.allclose(1.00, np.sum(cp.asnumpy(
        cu_shap_values)) / (fx - expected), rtol=1e-01)

    if not close_values:
        print("cu_shap_values: ")
        print(cu_shap_values)
        print("golden_result_values")
        print(golden_result_values)

    assert expected_sum
    assert close_values

###############################################################################
#                              End to end tests                               #
###############################################################################


@pytest.mark.parametrize("model", [cuml.LinearRegression,
                                   cuml.KNeighborsRegressor,
                                   cuml.SVR])
def test_exact_regression_datasets(exact_tests_dataset, model):
    # todo (dd): idx parameter is for repeating the test for a few CI runs
    # will be removed before merging
    X_train, X_test, y_train, y_test = exact_tests_dataset

    mod = model().fit(X_train, y_train)

    explainer = cuml.experimental.explainer.KernelExplainer(
        model=mod.predict,
        data=X_train)

    cu_shap_values = explainer.shap_values(X_test)
    experimental_test_and_log(cu_shap_values,
                              golden_regression_results[model],
                              mod.predict(X_test),
                              float(explainer.expected_value))

    skmod = cuml_skl_class_dict[model]().fit(X_train, y_train)

    explainer = cuml.experimental.explainer.KernelExplainer(
        model=skmod.predict,
        data=X_train)

    cu_shap_values = explainer.shap_values(X_test)

    # since the values were calculated with the cuml models, a little
    # looser tolerance in the comparison is expected
    experimental_test_and_log(cu_shap_values,
                              golden_regression_results[model],
                              mod.predict(X_test),
                              float(explainer.expected_value))


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

    explainer = cuml.experimental.explainer.KernelExplainer(
        model=mod.predict_proba,
        data=X_train)

    cu_shap_values = explainer.shap_values(X_test)

    experimental_test_and_log(cu_shap_values[0],
                              golden_classification_result[0],
                              float(mod.predict_proba(X_test)[0][0]),
                              float(explainer.expected_value[0]),
                              tolerance=1e-01)

    experimental_test_and_log(cu_shap_values[1],
                              golden_classification_result[1],
                              float(mod.predict_proba(X_test)[0][1]),
                              float(explainer.expected_value[1]),
                              tolerance=1e-01)

    mod = sklearn.svm.SVC(probability=True).fit(X_train, y_train)

    explainer = cuml.experimental.explainer.KernelExplainer(
        model=mod.predict_proba,
        data=X_train)

    cu_shap_values = explainer.shap_values(X_test)

    # Some values are very small, which mean our tolerance here needs to be
    # a little looser to avoid false positives from comparisons like
    # 0.00348627 - 0.00247397. The loose tolerance still tests that the
    # distribution of the values matches.
    experimental_test_and_log(cu_shap_values[0],
                              golden_classification_result[0],
                              float(mod.predict_proba(X_test)[0][0]),
                              float(explainer.expected_value[0]),
                              tolerance=1e-01)

    experimental_test_and_log(cu_shap_values[1],
                              golden_classification_result[1],
                              float(mod.predict_proba(X_test)[0][1]),
                              float(explainer.expected_value[1]),
                              tolerance=1e-01)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("nfeatures", [10])
@pytest.mark.parametrize("nbackground", [10])
@pytest.mark.parametrize("model", [cuml.TruncatedSVD,
                                   cuml.PCA])
def test_kernel_shap_standalone(dtype, nfeatures, nbackground, model):
    X, y = cuml.datasets.make_regression(n_samples=nbackground + 10,
                                         n_features=nfeatures,
                                         noise=0.1, dtype=dtype)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=2)

    mod = model(n_components=3).fit(X_train, y_train)

    cu_explainer = \
        cuml.experimental.explainer.KernelExplainer(model=mod.transform,
                                                    data=X_train,
                                                    is_gpu_model=True)

    cu_shap_values = cu_explainer.shap_values(X_test)
    exp_v = cu_explainer.expected_value

    # we have 5 lists of shap values, each corresponding to a component since
    # transform gives back arrays of shape (nrows x ncomponents)
    # we test that for each test row, for each component, the
    # sum of the shap values is the same as the difference between the
    # expected value for that component minus the value of the transform of
    # the row.
    for sv_idx in range(2):
        # pca and tsvd transform give results back nested
        fx = mod.transform(X_test[sv_idx].reshape(1, nfeatures))[0]

        for comp_idx in range(3):
            assert(
                np.sum(
                    cu_shap_values[comp_idx][sv_idx]) - abs(
                        fx[comp_idx] - exp_v[comp_idx])) <= 1e-5


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("nfeatures", [11, 15])
@pytest.mark.parametrize("nbackground", [30])
@pytest.mark.parametrize("model", [cuml.SVR])
def test_kernel_gpu_cpu_shap(dtype, nfeatures, nbackground, model):
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
        cuml.experimental.explainer.KernelExplainer(model=mod.predict,
                                                    data=X_train,
                                                    is_gpu_model=True)

    cu_shap_values = cu_explainer.shap_values(X_test)

    exp_v = cu_explainer._expected_value
    fx = mod.predict(X_test)
    for test_idx in range(5):
        assert(np.sum(
            cu_shap_values[test_idx]) - abs(fx[test_idx] - exp_v)) <= 1e-5

    if has_shap("0.37"):
        import shap
        explainer = shap.KernelExplainer(mod.predict, cp.asnumpy(X_train))
        shap_values = explainer.shap_values(cp.asnumpy(X_test))

        # note that small variances in the l1_regression with larger
        # n_features, even among runs of the same explainer can cause this
        # test to be flaky, better testing strategy in process.
        assert np.allclose(cu_shap_values, shap_values, rtol=1e-01, atol=1e-01)


def test_kernel_housing_dataset(housing_dataset):
    X, y, _ = housing_dataset

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # making all float32 to use gpu predict on random forest
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    cumodel = cuml.RandomForestRegressor().fit(X_train, y_train)

    explainer = cuml.experimental.explainer.KernelExplainer(
        model=cumodel.predict,
        data=X_train[:100],
        output_type='numpy')

    cu_shap_values = explainer.shap_values(X_test[:2])

    assert np.allclose(cu_shap_values, housing_regression_result,
                       rtol=1e-01, atol=1e-01)

    assert True

###############################################################################
#                        Single function unit tests                           #
###############################################################################


def test_binom_coef():
    for i in range(1, 101):
        val = cuml.experimental.explainer.kernel_shap._binomCoef(100, i)
        if has_scipy():
            from scipy.special import binom
            assert math.isclose(val, binom(100, i), rel_tol=1e-15)


def test_shapley_kernel():
    for i in range(11):
        val = cuml.experimental.explainer.kernel_shap._shapley_kernel(10, i)
        assert val == shapley_kernel_results[i]


def test_full_powerset():
    ps, w = cuml.experimental.explainer.kernel_shap._powerset(
        5, 2, 2**5 - 2, full_powerset=True)

    for i in range(len(ps)):
        assert np.all(ps[i] == full_powerset_result[i])
        assert math.isclose(w[i], full_powerset_weight_result[i])


def test_partial_powerset():
    ps, w = cuml.experimental.explainer.kernel_shap._powerset(6, 3, 42)

    for i in range(len(ps)):
        assert np.all(ps[i] == partial_powerset_result[i])
        assert math.isclose(w[i], partial_powerset_weight_result[i])


@pytest.mark.parametrize("full_powerset", [True, False])
def test_get_number_of_exact_random_samples(full_powerset):

    if full_powerset:
        nsamples_exact, nsamples_random, ind = \
            (cuml.experimental.explainer.kernel_shap.
             _get_number_of_exact_random_samples(10, 2**10 + 1))
        assert nsamples_exact == 1022
        assert nsamples_random == 0
        assert ind == 5
    else:
        nsamples_exact, nsamples_random, ind = \
            (cuml.experimental.explainer.kernel_shap.
             _get_number_of_exact_random_samples(10, 100))

        assert nsamples_exact == 20
        assert nsamples_random == 80
        assert ind == 2


def test_generate_nsamples_weights():
    samples, w = \
        cuml.experimental.explainer.kernel_shap._generate_nsamples_weights(
            ncols=20,
            nsamples=30,
            nsamples_exact=10,
            nsamples_random=20,
            randind=5,
            dtype=np.float32
        )
    # check that all our samples are between 5 and 6, and the weights in pairs
    # are generated correctly
    for i, s in enumerate(samples):
        assert s in [5, 6]
        assert w[i * 2] == \
            cuml.experimental.explainer.kernel_shap._shapley_kernel(20, int(s))
        assert w[i * 2 + 1] == \
            cuml.experimental.explainer.kernel_shap._shapley_kernel(20, int(s))


@pytest.mark.parametrize("l1_type", ['auto', 'aic', 'bic', 'num_features(3)',
                                     0.2])
def test_l1_regularization(exact_tests_dataset, l1_type):
    # currently this is a code test, not mathematical results test.
    # Hard to test without falling into testing the underlying algorithms
    # which are out of this unit test scope.
    X, w = cuml.experimental.explainer.kernel_shap._powerset(
        5, 2, 2**5 - 2, full_powerset=True)

    y = cp.random.rand(X.shape[0])
    nz = \
        cuml.experimental.explainer.kernel_shap._l1_regularization(
            X=cp.asarray(X).astype(np.float32),
            y=cp.asarray(y).astype(np.float32),
            weights=cp.asarray(w),
            expected_value=0.0,
            fx=0.0,
            link_fn=cuml.experimental.explainer.common.identity,
            l1_reg=l1_type
        )
    assert isinstance(nz, cp.ndarray)


###############################################################################
#                                 Precomputed results                         #
#                               and testing variables                         #
###############################################################################

# "golden" results obtained by running brute force Kernel SHAP notebook from
# https://github.com/slundberg/shap/blob/master/notebooks/kernel_explainer/Simple%20Kernel%20SHAP.ipynb
# and confirmed with SHAP package.
golden_regression_results = {
    cuml.LinearRegression: [
        -3.6001968e-01, -1.0214063e+02, 1.2992077e+00, -6.3079113e+01,
        2.5177002e-04, -2.3135548e+00, -1.0176431e+02, 3.3992329e+00,
        4.1034698e+01, 7.1334076e+01, -1.6048431e+00
    ],
    cuml.KNeighborsRegressor: [
        3.3001919, -46.435326, -5.2908664, -34.01667, -5.917948, -14.939089,
        -46.88066, -3.1448324, 11.431797, 49.297226, 5.9906464
    ],
    cuml.SVR: [
        0.04022658, -1.019261, 0.03412837, -0.7708928, -0.01342008,
        -0.10700871, -1.2565054, 0.49404335, 0.4250477, 1.0444777, 0.01112604
    ]
}

# For testing predict proba, we get one array of shap values per class
golden_classification_result = [
    [0.00152159, 0.00247397, 0.00250474, 0.00155965, 0.0113184,
     -0.01153999, 0.19297145, 0.17027254, 0.00850102, -0.01293354,
     -0.00088981],
    [-0.00152159, -0.00247397, -0.00250474, -0.00155965, -0.0113184,
     0.01153999, -0.19297145, -0.17027254, -0.00850102, 0.01293354,
     0.00088981]
]

housing_regression_result = np.array(
    [[-0.7222878, 0.00888237, -0.07044561, -0.02764106, -0.01486777,
      -0.19961227, -0.1367276, -0.11073875],
     [-0.688218, 0.04260924, -0.12853414, 0.06109668, -0.01486243,
      -0.0627693, -0.17290883, -0.02488524]], dtype=np.float32)

cuml_skl_class_dict = {
    cuml.LinearRegression: sklearn.linear_model.LinearRegression,
    cuml.KNeighborsRegressor: sklearn.neighbors.KNeighborsRegressor,
    cuml.SVR: sklearn.svm.SVR
}


# results for individual function unit tests
shapley_kernel_results = [10000, 0.1, 0.0125, 0.0035714285714285713,
                          0.0017857142857142857, 0.0014285714285714286,
                          0.0017857142857142857, 0.0035714285714285713,
                          0.0125, 0.1, 10000]

full_powerset_result = [[1., 0., 0., 0., 0.],
                        [0., 1., 0., 0., 0.],
                        [0., 0., 1., 0., 0.],
                        [0., 0., 0., 1., 0.],
                        [0., 0., 0., 0., 1.],
                        [1., 1., 0., 0., 0.],
                        [1., 0., 1., 0., 0.],
                        [1., 0., 0., 1., 0.],
                        [1., 0., 0., 0., 1.],
                        [0., 1., 1., 0., 0.],
                        [0., 1., 0., 1., 0.],
                        [0., 1., 0., 0., 1.],
                        [0., 0., 1., 1., 0.],
                        [0., 0., 1., 0., 1.],
                        [0., 0., 0., 1., 1.],
                        [1., 1., 1., 0., 0.],
                        [1., 1., 0., 1., 0.],
                        [1., 1., 0., 0., 1.],
                        [1., 0., 1., 1., 0.],
                        [1., 0., 1., 0., 1.],
                        [1., 0., 0., 1., 1.],
                        [0., 1., 1., 1., 0.],
                        [0., 1., 1., 0., 1.],
                        [0., 1., 0., 1., 1.],
                        [0., 0., 1., 1., 1.],
                        [1., 1., 1., 1., 0.],
                        [1., 1., 1., 0., 1.],
                        [1., 1., 0., 1., 1.],
                        [1., 0., 1., 1., 1.],
                        [0., 1., 1., 1., 1.]]


full_powerset_weight_result = np.array(
    [0.2, 0.2, 0.2, 0.2, 0.2, 0.06666667, 0.06666667, 0.06666667, 0.06666667,
     0.06666667, 0.06666667, 0.06666667, 0.06666667, 0.06666667, 0.06666667,
     0.06666667, 0.06666667, 0.06666667, 0.06666667, 0.06666667, 0.06666667,
     0.06666667, 0.06666667, 0.06666667, 0.06666667, 0.2, 0.2, 0.2, 0.2, 0.2],
    dtype=np.float32
)

partial_powerset_result = [[1., 0., 0., 0., 0., 0.],
                           [0., 1., 1., 1., 1., 1.],
                           [0., 1., 0., 0., 0., 0.],
                           [1., 0., 1., 1., 1., 1.],
                           [0., 0., 1., 0., 0., 0.],
                           [1., 1., 0., 1., 1., 1.],
                           [0., 0., 0., 1., 0., 0.],
                           [1., 1., 1., 0., 1., 1.],
                           [0., 0., 0., 0., 1., 0.],
                           [1., 1., 1., 1., 0., 1.],
                           [0., 0., 0., 0., 0., 1.],
                           [1., 1., 1., 1., 1., 0.],
                           [1., 1., 0., 0., 0., 0.],
                           [0., 0., 1., 1., 1., 1.],
                           [1., 0., 1., 0., 0., 0.],
                           [0., 1., 0., 1., 1., 1.],
                           [1., 0., 0., 1., 0., 0.],
                           [0., 1., 1., 0., 1., 1.],
                           [1., 0., 0., 0., 1., 0.],
                           [0., 1., 1., 1., 0., 1.],
                           [1., 0., 0., 0., 0., 1.],
                           [0., 1., 1., 1., 1., 0.],
                           [0., 1., 1., 0., 0., 0.],
                           [1., 0., 0., 1., 1., 1.],
                           [0., 1., 0., 1., 0., 0.],
                           [1., 0., 1., 0., 1., 1.],
                           [0., 1., 0., 0., 1., 0.],
                           [1., 0., 1., 1., 0., 1.],
                           [0., 1., 0., 0., 0., 1.],
                           [1., 0., 1., 1., 1., 0.],
                           [0., 0., 1., 1., 0., 0.],
                           [1., 1., 0., 0., 1., 1.],
                           [0., 0., 1., 0., 1., 0.],
                           [1., 1., 0., 1., 0., 1.],
                           [0., 0., 1., 0., 0., 1.],
                           [1., 1., 0., 1., 1., 0.],
                           [0., 0., 0., 1., 1., 0.],
                           [1., 1., 1., 0., 0., 1.],
                           [0., 0., 0., 1., 0., 1.],
                           [1., 1., 1., 0., 1., 0.],
                           [0., 0., 0., 0., 1., 1.],
                           [1., 1., 1., 1., 0., 0.]]

partial_powerset_weight_result = np.array(
    [0.16666667, 0.16666667, 0.16666667, 0.16666667,
     0.16666667, 0.16666667, 0.16666667, 0.16666667,
     0.16666667, 0.16666667, 0.16666667, 0.16666667,
     0.041666668, 0.041666668, 0.041666668, 0.041666668,
     0.041666668, 0.041666668, 0.041666668, 0.041666668,
     0.041666668, 0.041666668, 0.041666668, 0.041666668,
     0.041666668, 0.041666668, 0.041666668, 0.041666668,
     0.041666668, 0.041666668, 0.041666668, 0.041666668,
     0.041666668, 0.041666668, 0.041666668, 0.041666668,
     0.041666668, 0.041666668, 0.041666668, 0.041666668,
     0.041666668, 0.041666668], dtype=np.float32)

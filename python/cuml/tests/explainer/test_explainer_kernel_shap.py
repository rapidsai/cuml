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

import math

import cupy as cp
import numpy as np
import pytest
import scipy.special
import sklearn.neighbors
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

import cuml
from cuml import KernelExplainer, Lasso
from cuml.datasets import make_regression
from cuml.testing.datasets import with_dtype
from cuml.testing.utils import ClassEnumerator, get_shap_values

models_config = ClassEnumerator(module=cuml)
models = models_config.get_models()


def assert_and_log(
    cu_shap_values, golden_result_values, fx, expected, tolerance=1e-02
):
    close_values = np.allclose(
        cu_shap_values, golden_result_values, rtol=tolerance, atol=tolerance
    )

    expected_sum = np.allclose(
        1.00, np.sum(cp.asnumpy(cu_shap_values)) / (fx - expected), rtol=1e-01
    )

    if not close_values:
        print("cu_shap_values: ")
        print(cu_shap_values)
        print("golden_result_values")
        print(golden_result_values)

    if not expected_sum:
        print(np.sum(cp.asnumpy(cu_shap_values)))

    assert expected_sum
    assert close_values


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
        for i in range(3):
            print(i)
            assert_and_log(
                shap_values[i],
                golden_regression_results[model][i],
                mod.predict(X_test[i].reshape(1, X_test.shape[1])),
                explainer.expected_value,
            )


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
            explainer=KernelExplainer,
        )

        # Some values are very small, which mean our tolerance here needs to be
        # a little looser to avoid false positives from comparisons like
        # 0.00348627 - 0.00247397. The loose tolerance still tests that the
        # distribution of the values matches.
        for idx, svs in enumerate(shap_values):
            assert_and_log(
                svs[0],
                golden_classification_result[idx],
                float(mod.predict_proba(X_test)[0][idx]),
                explainer.expected_value[idx],
                tolerance=1e-01,
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

    # we have 3 lists of shap values, each corresponding to a component since
    # transform gives back arrays of shape (nrows x ncomponents)
    # we test that for each test row, for each component, the
    # sum of the shap values is the same as the difference between the
    # expected value for that component minus the value of the transform of
    # the row.
    for sv_idx in range(3):
        # pca and tsvd transform give results back nested
        fx = mod.transform(X_test[sv_idx].reshape(1, n_features))[0]

        for comp_idx in range(3):
            assert (
                np.sum(shap_values[comp_idx][sv_idx])
                - abs(fx[comp_idx] - exp_v[comp_idx])
            ) <= 1e-5


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
    for test_idx in range(3):
        assert (
            np.sum(shap_values[test_idx]) - abs(fx[test_idx] - exp_v)
        ) <= 1e-5

    explainer = shap.KernelExplainer(mod.predict, cp.asnumpy(X_train))
    cpu_shap_values = explainer.shap_values(cp.asnumpy(X_test))

    assert np.allclose(shap_values, cpu_shap_values, rtol=1e-01, atol=1e-01)


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

    cumodel = cuml.RandomForestRegressor(max_features="sqrt").fit(
        X_train, y_train
    )

    explainer = KernelExplainer(
        model=cumodel.predict, data=X_train[:100], output_type="numpy"
    )

    cu_shap_values = explainer.shap_values(X_test[:2])

    assert np.allclose(
        cu_shap_values, housing_regression_result, rtol=1e-01, atol=1e-01
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
        assert val == shapley_kernel_results[i]


def test_full_powerset():
    ps, w = cuml.explainer.kernel_shap._powerset(
        5, 2, 2**5 - 2, full_powerset=True
    )

    for i in range(len(ps)):
        assert np.all(ps[i] == full_powerset_result[i])
        assert math.isclose(w[i], full_powerset_weight_result[i])


def test_partial_powerset():
    ps, w = cuml.explainer.kernel_shap._powerset(6, 3, 42)

    for i in range(len(ps)):
        assert np.all(ps[i] == partial_powerset_result[i])
        assert math.isclose(w[i], partial_powerset_weight_result[i])


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
        assert w[i * 2] == cuml.explainer.kernel_shap._shapley_kernel(
            20, int(s)
        )
        assert w[i * 2 + 1] == cuml.explainer.kernel_shap._shapley_kernel(
            20, int(s)
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
        if "operands could not be broadcast together" in str(error):
            pytest.xfail(
                "Known sklearn LARS broadcasting bug - see scikit-learn#9603"
            )
        else:
            raise error


###############################################################################
#                                 Precomputed results                         #
#                               and testing variables                         #
###############################################################################

# "golden" results obtained by running brute force Kernel SHAP notebook from
# https://github.com/slundberg/shap/blob/master/notebooks/kernel_explainer/Simple%20Kernel%20SHAP.ipynb
# and confirmed with SHAP package.
golden_regression_results = {
    cuml.LinearRegression: [
        [
            -1.3628216e00,
            -1.0234555e02,
            1.3433075e-01,
            -6.1763966e01,
            2.6035309e-04,
            -3.4455872e00,
            -1.0159061e02,
            3.4058199e00,
            4.1598396e01,
            7.2152481e01,
            -2.1964417e00,
        ],
        [
            -8.6558792e01,
            8.9456577e00,
            -3.6405910e01,
            1.0574381e01,
            -4.1580200e-04,
            -5.8939896e01,
            4.8407948e01,
            1.4475842e00,
            -2.0742226e01,
            6.6378265e01,
            -3.5134201e01,
        ],
        [
            -1.3722158e01,
            -2.9430325e01,
            -8.0079269e01,
            1.2096907e02,
            1.0681152e-03,
            -5.4266449e01,
            -3.1012087e01,
            -7.9640961e-01,
            7.7072838e01,
            1.5370981e01,
            -2.4032040e01,
        ],
    ],
    cuml.KNeighborsRegressor: [
        [
            4.3210926,
            -47.497078,
            -4.523407,
            -35.49657,
            -5.5174675,
            -14.158726,
            -51.303787,
            -2.6457424,
            12.230529,
            52.345207,
            6.3014755,
        ],
        [
            -52.036957,
            2.4158602,
            -20.302296,
            15.428952,
            5.9823637,
            -20.046719,
            22.46046,
            -4.762917,
            -6.20145,
            37.457417,
            5.3511925,
        ],
        [
            -8.803419,
            -7.4095736,
            -48.113777,
            57.21296,
            1.0490589,
            -37.94751,
            -20.748789,
            -0.22258139,
            28.204493,
            4.5492225,
            0.5797138,
        ],
    ],
    cuml.SVR: [
        [
            3.53810340e-02,
            -8.11021507e-01,
            3.34369540e-02,
            -8.68727207e-01,
            1.06804073e-03,
            -1.14741415e-01,
            -1.35545099e00,
            3.87545109e-01,
            4.43311602e-01,
            1.08623052e00,
            2.65314579e-02,
        ],
        [
            -1.39247358e00,
            5.91157824e-02,
            -4.33764964e-01,
            1.04503572e-01,
            -4.41753864e-03,
            -1.09017754e00,
            5.90143979e-01,
            1.08445108e-01,
            -2.26831138e-01,
            9.69056726e-01,
            -1.18437767e-01,
        ],
        [
            -1.28573015e-01,
            -2.33658075e-01,
            -1.02735841e00,
            1.47447693e00,
            -1.99043751e-03,
            -1.11328888e00,
            -4.66209412e-01,
            -1.02243885e-01,
            8.18460345e-01,
            2.20144764e-01,
            -9.62769389e-02,
        ],
    ],
}

# For testing predict proba, we get one array of shap values per class
golden_classification_result = [
    [
        0.00152159,
        0.00247397,
        0.00250474,
        0.00155965,
        0.0113184,
        -0.01153999,
        0.19297145,
        0.17027254,
        0.00850102,
        -0.01293354,
        -0.00088981,
    ],
    [
        -0.00152159,
        -0.00247397,
        -0.00250474,
        -0.00155965,
        -0.0113184,
        0.01153999,
        -0.19297145,
        -0.17027254,
        -0.00850102,
        0.01293354,
        0.00088981,
    ],
]

housing_regression_result = np.array(
    [
        [
            -0.00182223,
            -0.01232004,
            -0.4782278,
            0.04781425,
            -0.01337761,
            -0.34830606,
            -0.4682865,
            -0.20812261,
        ],
        [
            -0.0013606,
            0.0110372,
            -0.445176,
            -0.08268094,
            0.00406259,
            -0.02185595,
            -0.47673094,
            -0.13557231,
        ],
    ],
    dtype=np.float32,
)

cuml_skl_class_dict = {
    cuml.LinearRegression: sklearn.linear_model.LinearRegression,
    cuml.KNeighborsRegressor: sklearn.neighbors.KNeighborsRegressor,
    cuml.SVR: sklearn.svm.SVR,
}


# results for individual function unit tests
shapley_kernel_results = [
    10000,
    0.1,
    0.0125,
    0.0035714285714285713,
    0.0017857142857142857,
    0.0014285714285714286,
    0.0017857142857142857,
    0.0035714285714285713,
    0.0125,
    0.1,
    10000,
]

full_powerset_result = [
    [1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0, 1.0, 0.0],
    [1.0, 0.0, 1.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 1.0, 1.0],
    [0.0, 1.0, 1.0, 1.0, 0.0],
    [0.0, 1.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 0.0],
    [1.0, 1.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0, 1.0, 1.0],
]


full_powerset_weight_result = np.array(
    [
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
        0.06666667,
        0.06666667,
        0.06666667,
        0.06666667,
        0.06666667,
        0.06666667,
        0.06666667,
        0.06666667,
        0.06666667,
        0.06666667,
        0.06666667,
        0.06666667,
        0.06666667,
        0.06666667,
        0.06666667,
        0.06666667,
        0.06666667,
        0.06666667,
        0.06666667,
        0.06666667,
        0.2,
        0.2,
        0.2,
        0.2,
        0.2,
    ],
    dtype=np.float32,
)

partial_powerset_result = [
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [1.0, 1.0, 1.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 1.0, 1.0, 1.0],
    [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    [0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
    [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
    [0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
    [0.0, 1.0, 0.0, 1.0, 0.0, 0.0],
    [1.0, 0.0, 1.0, 0.0, 1.0, 1.0],
    [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    [1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    [1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 1.0],
    [1.0, 1.0, 0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
    [1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
]

partial_powerset_weight_result = np.array(
    [
        0.16666667,
        0.16666667,
        0.16666667,
        0.16666667,
        0.16666667,
        0.16666667,
        0.16666667,
        0.16666667,
        0.16666667,
        0.16666667,
        0.16666667,
        0.16666667,
        0.041666668,
        0.041666668,
        0.041666668,
        0.041666668,
        0.041666668,
        0.041666668,
        0.041666668,
        0.041666668,
        0.041666668,
        0.041666668,
        0.041666668,
        0.041666668,
        0.041666668,
        0.041666668,
        0.041666668,
        0.041666668,
        0.041666668,
        0.041666668,
        0.041666668,
        0.041666668,
        0.041666668,
        0.041666668,
        0.041666668,
        0.041666668,
        0.041666668,
        0.041666668,
        0.041666668,
        0.041666668,
        0.041666668,
        0.041666668,
    ],
    dtype=np.float32,
)

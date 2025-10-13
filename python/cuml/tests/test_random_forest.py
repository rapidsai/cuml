# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

import json
import math
import os
import random
import warnings

import cudf
import numpy as np
import pytest
import treelite
from cudf.pandas import LOADED as cudf_pandas_active
from numba import cuda
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_iris,
    make_classification,
    make_regression,
)
from sklearn.ensemble import RandomForestClassifier as skrfc
from sklearn.ensemble import RandomForestRegressor as skrfr
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    mean_tweedie_deviance,
)
from sklearn.model_selection import train_test_split

import cuml
import cuml.internals.logger as logger
from cuml.ensemble import RandomForestClassifier as curfc
from cuml.ensemble import RandomForestRegressor as curfr
from cuml.ensemble.randomforest_common import compute_max_features
from cuml.metrics import r2_score
from cuml.testing.utils import (
    get_handle,
    quality_param,
    stress_param,
    unit_param,
)


@pytest.fixture(
    scope="session",
    params=[
        unit_param({"n_samples": 350, "n_features": 20, "n_informative": 10}),
        quality_param(
            {"n_samples": 5000, "n_features": 200, "n_informative": 80}
        ),
        stress_param(
            {"n_samples": 500000, "n_features": 400, "n_informative": 180}
        ),
    ],
)
def small_clf(request):
    X, y = make_classification(
        n_samples=request.param["n_samples"],
        n_features=request.param["n_features"],
        n_clusters_per_class=1,
        n_informative=request.param["n_informative"],
        random_state=123,
        n_classes=2,
    )
    return X, y


@pytest.fixture(
    scope="session",
    params=[
        unit_param({"n_samples": 350, "n_features": 30, "n_informative": 15}),
        quality_param(
            {"n_samples": 5000, "n_features": 200, "n_informative": 80}
        ),
        stress_param(
            {"n_samples": 500000, "n_features": 400, "n_informative": 180}
        ),
    ],
)
def mclass_clf(request):
    X, y = make_classification(
        n_samples=request.param["n_samples"],
        n_features=request.param["n_features"],
        n_clusters_per_class=1,
        n_informative=request.param["n_informative"],
        random_state=123,
        n_classes=10,
    )
    return X, y


@pytest.fixture(
    scope="session",
    params=[
        unit_param({"n_samples": 500, "n_features": 20, "n_informative": 10}),
        quality_param(
            {"n_samples": 5000, "n_features": 200, "n_informative": 50}
        ),
        stress_param(
            {"n_samples": 500000, "n_features": 400, "n_informative": 100}
        ),
    ],
)
def large_clf(request):
    X, y = make_classification(
        n_samples=request.param["n_samples"],
        n_features=request.param["n_features"],
        n_clusters_per_class=1,
        n_informative=request.param["n_informative"],
        random_state=123,
        n_classes=2,
    )
    return X, y


@pytest.fixture(
    scope="session",
    params=[
        unit_param({"n_samples": 1500, "n_features": 20, "n_informative": 10}),
        quality_param(
            {"n_samples": 12000, "n_features": 200, "n_informative": 100}
        ),
        stress_param(
            {"n_samples": 500000, "n_features": 500, "n_informative": 350}
        ),
    ],
)
def large_reg(request):
    X, y = make_regression(
        n_samples=request.param["n_samples"],
        n_features=request.param["n_features"],
        n_informative=request.param["n_informative"],
        random_state=123,
    )
    return X, y


special_reg_params = [
    unit_param(
        {
            "mode": "unit",
            "n_samples": 500,
            "n_features": 20,
            "n_informative": 10,
        }
    ),
    quality_param(
        {
            "mode": "quality",
            "n_samples": 500,
            "n_features": 20,
            "n_informative": 10,
        }
    ),
    quality_param({"mode": "quality", "n_features": 200, "n_informative": 50}),
    stress_param(
        {
            "mode": "stress",
            "n_samples": 500,
            "n_features": 20,
            "n_informative": 10,
        }
    ),
    stress_param({"mode": "stress", "n_features": 200, "n_informative": 50}),
    stress_param(
        {
            "mode": "stress",
            "n_samples": 1000,
            "n_features": 400,
            "n_informative": 100,
        }
    ),
]


@pytest.fixture(scope="session", params=special_reg_params)
def special_reg(request):
    if request.param["mode"] == "quality":
        X, y = fetch_california_housing(return_X_y=True)
    else:
        X, y = make_regression(
            n_samples=request.param["n_samples"],
            n_features=request.param["n_features"],
            n_informative=request.param["n_informative"],
            random_state=123,
        )
    return X, y


def test_default_parameters():
    reg_params = curfr().get_params()
    clf_params = curfc().get_params()

    # Different default max_features
    assert reg_params["max_features"] == 1.0
    assert clf_params["max_features"] == "sqrt"

    # Different default split_criterion
    assert reg_params["split_criterion"] == "mse"
    assert clf_params["split_criterion"] == "gini"

    # Drop differing params
    for name in [
        "max_features",
        "split_criterion",
        "handle",
    ]:
        reg_params.pop(name)
        clf_params.pop(name)

    # The rest are the same
    assert reg_params == clf_params


@pytest.mark.parametrize("max_depth", [2, 4])
@pytest.mark.parametrize(
    "split_criterion", ["poisson", "gamma", "inverse_gaussian"]
)
def test_tweedie_convergence(max_depth, split_criterion):
    np.random.seed(33)
    bootstrap = None
    max_features = 1.0
    n_estimators = 1
    min_impurity_decrease = 1e-5
    n_datapoints = 1000
    tweedie = {
        "poisson": {"power": 1, "gen": np.random.poisson, "args": [0.01]},
        "gamma": {"power": 2, "gen": np.random.gamma, "args": [2.0]},
        "inverse_gaussian": {
            "power": 3,
            "gen": np.random.wald,
            "args": [0.1, 2.0],
        },
    }
    # generating random dataset with tweedie distribution
    X = np.random.random((n_datapoints, 4)).astype(np.float32)
    y = tweedie[split_criterion]["gen"](
        *tweedie[split_criterion]["args"], size=n_datapoints
    ).astype(np.float32)

    tweedie_preds = (
        curfr(
            split_criterion=split_criterion,
            max_depth=max_depth,
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
        )
        .fit(X, y)
        .predict(X)
    )
    mse_preds = (
        curfr(
            split_criterion=2,
            max_depth=max_depth,
            n_estimators=n_estimators,
            bootstrap=bootstrap,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
        )
        .fit(X, y)
        .predict(X)
    ).squeeze()
    # y should not be non-positive for mean_poisson_deviance
    mask = mse_preds > 0
    mse_tweedie_deviance = mean_tweedie_deviance(
        y[mask], mse_preds[mask], power=tweedie[split_criterion]["power"]
    )
    tweedie_tweedie_deviance = mean_tweedie_deviance(
        y[mask], tweedie_preds[mask], power=tweedie[split_criterion]["power"]
    )

    # model trained on tweedie data with
    # tweedie criterion must perform better on tweedie loss
    assert mse_tweedie_deviance >= tweedie_tweedie_deviance


@pytest.mark.parametrize(
    "max_samples", [unit_param(1.0), quality_param(0.90), stress_param(0.95)]
)
@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("max_features", [1.0, "log2", "sqrt"])
@pytest.mark.skipif(
    cudf_pandas_active,
    reason="cudf.pandas causes sklearn RF estimators crashes sometimes. "
    "Issue: https://github.com/rapidsai/cuml/issues/5991",
)
def test_rf_classification(small_clf, datatype, max_samples, max_features):
    use_handle = True

    X, y = small_clf
    X = X.astype(datatype)
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )
    # Create a handle for the cuml model
    handle, stream = get_handle(use_handle, n_streams=1)

    # Initialize, fit and predict using cuML's
    # random forest classification model
    cuml_model = curfc(
        max_features=max_features,
        max_samples=max_samples,
        n_bins=16,
        split_criterion=0,
        min_samples_leaf=2,
        random_state=123,
        n_streams=1,
        n_estimators=40,
        handle=handle,
        max_leaves=-1,
        max_depth=16,
    )
    cuml_model.fit(X_train, y_train)

    preds = cuml_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    if X.shape[0] < 500000:
        sk_model = skrfc(
            n_estimators=40,
            max_depth=16,
            min_samples_split=2,
            max_features=max_features,
            random_state=10,
        )
        sk_model.fit(X_train, y_train)
        sk_preds = sk_model.predict(X_test)
        sk_acc = accuracy_score(y_test, sk_preds)
        assert acc >= (sk_acc - 0.07)


@pytest.mark.parametrize(
    "max_samples", [unit_param(1.0), quality_param(0.90), stress_param(0.95)]
)
@pytest.mark.parametrize("datatype", [np.float32, np.float64])
def test_rf_classification_unorder(
    small_clf, datatype, max_samples, max_features=1, a=2, b=5
):
    use_handle = True

    X, y = small_clf
    X = X.astype(datatype)
    y = y.astype(np.int32)
    # affine transformation
    y = a * y + b
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )
    # Create a handle for the cuml model
    handle, stream = get_handle(use_handle, n_streams=1)

    # Initialize, fit and predict using cuML's
    # random forest classification model
    cuml_model = curfc(
        max_features=max_features,
        max_samples=max_samples,
        n_bins=16,
        split_criterion=0,
        min_samples_leaf=2,
        random_state=123,
        n_streams=1,
        n_estimators=40,
        handle=handle,
        max_leaves=-1,
        max_depth=16,
    )
    cuml_model.fit(X_train, y_train)

    preds = cuml_model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    if X.shape[0] < 500000:
        sk_model = skrfc(
            n_estimators=40,
            max_depth=16,
            min_samples_split=2,
            max_features=max_features,
            random_state=10,
        )
        sk_model.fit(X_train, y_train)
        sk_preds = sk_model.predict(X_test)
        sk_acc = accuracy_score(y_test, sk_preds)
        assert acc >= (sk_acc - 0.07)


@pytest.mark.parametrize(
    "max_samples", [unit_param(1.0), quality_param(0.90), stress_param(0.95)]
)
@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "max_features,n_bins",
    [
        (1.0, 16),
        (1.0, 11),
        ("log2", 100),
        ("sqrt", 100),
        (1.0, 17),
        (1.0, 32),
    ],
)
@pytest.mark.skipif(
    cudf_pandas_active,
    reason="cudf.pandas causes sklearn RF estimators crashes sometimes. "
    "Issue: https://github.com/rapidsai/cuml/issues/5991",
)
def test_rf_regression(
    special_reg, datatype, max_features, max_samples, n_bins
):

    use_handle = True

    X, y = special_reg
    X = X.astype(datatype)
    y = y.astype(datatype)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )

    # Create a handle for the cuml model
    handle, stream = get_handle(use_handle, n_streams=1)

    # Initialize and fit using cuML's random forest regression model
    cuml_model = curfr(
        max_features=max_features,
        max_samples=max_samples,
        n_bins=n_bins,
        split_criterion=2,
        min_samples_leaf=2,
        random_state=123,
        n_streams=1,
        n_estimators=50,
        handle=handle,
        max_leaves=-1,
        max_depth=16,
    )
    cuml_model.fit(X_train, y_train)
    preds = cuml_model.predict(X_test)

    r2 = r2_score(y_test, preds)
    # Initialize, fit and predict using sklearn's random forest regression model
    if X.shape[0] < 1000:  # mode != "stress"
        sk_model = skrfr(
            n_estimators=50,
            max_depth=16,
            min_samples_split=2,
            max_features=max_features,
            random_state=10,
        )
        sk_model.fit(X_train, y_train)
        sk_preds = sk_model.predict(X_test)
        sk_r2 = r2_score(y_test, sk_preds)
        assert r2 >= (sk_r2 - 0.07)


@pytest.mark.skipif(
    cudf_pandas_active,
    reason="cudf.pandas causes sklearn RF estimators crashes sometimes. "
    "Issue: https://github.com/rapidsai/cuml/issues/5991",
)
@pytest.mark.parametrize("datatype", [np.float32, np.float64])
def test_rf_classification_seed(small_clf, datatype):

    X, y = small_clf
    X = X.astype(datatype)
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )

    for i in range(8):
        seed = random.randint(100, 10**5)

        cu_class = curfc(random_state=seed, n_streams=1)
        cu_class.fit(X_train, y_train)
        preds_orig = cu_class.predict(X_test)
        acc_orig = accuracy_score(y_test, preds_orig)

        cu_class2 = curfc(random_state=seed, n_streams=1)
        cu_class2.fit(X_train, y_train)
        preds_rerun = cu_class2.predict(X_test)
        acc_rerun = accuracy_score(y_test, preds_rerun)

        assert acc_orig == acc_rerun
        assert (preds_orig == preds_rerun).all()


@pytest.mark.parametrize(
    "datatype", [(np.float64, np.float32), (np.float32, np.float64)]
)
@pytest.mark.parametrize("convert_dtype", [True, False])
@pytest.mark.filterwarnings("ignore:To use pickling(.*)::cuml[.*]")
@pytest.mark.skipif(
    cudf_pandas_active,
    reason="cudf.pandas causes sklearn RF estimators crashes sometimes. "
    "Issue: https://github.com/rapidsai/cuml/issues/5991",
)
def test_rf_classification_fit_and_predict_dtypes_differ(
    small_clf, datatype, convert_dtype
):
    X, y = small_clf
    X = X.astype(datatype[0])
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )
    X_test = X_test.astype(datatype[1])

    cuml_model = curfc()
    cuml_model.fit(X_train, y_train)
    preds = cuml_model.predict(X_test, convert_dtype=convert_dtype)
    acc = accuracy_score(y_test, preds)
    if X.shape[0] < 500000:
        sk_model = skrfc(max_depth=16, random_state=10)
        sk_model.fit(X_train, y_train)
        sk_preds = sk_model.predict(X_test)
        sk_acc = accuracy_score(y_test, sk_preds)
        assert acc >= (sk_acc - 0.07)


@pytest.mark.parametrize(
    "datatype", [(np.float64, np.float32), (np.float32, np.float64)]
)
@pytest.mark.filterwarnings("ignore:To use pickling(.*)::cuml[.*]")
@pytest.mark.skipif(
    cudf_pandas_active,
    reason="cudf.pandas causes sklearn RF estimators crashes sometimes. "
    "Issue: https://github.com/rapidsai/cuml/issues/5991",
)
def test_rf_regression_fit_and_predict_dtypes_differ(large_reg, datatype):
    X, y = large_reg
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )
    X_train = X_train.astype(datatype[0])
    y_train = y_train.astype(datatype[0])
    X_test = X_test.astype(datatype[1])
    y_test = y_test.astype(datatype[1])

    cuml_model = curfr()
    cuml_model.fit(X_train, y_train)
    preds = cuml_model.predict(X_test, convert_dtype=True)
    r2 = r2_score(y_test, preds)
    if X.shape[0] < 500000:
        sk_model = skrfr(max_depth=16, random_state=10)
        sk_model.fit(X_train, y_train)
        sk_preds = sk_model.predict(X_test)
        sk_r2 = r2_score(y_test, sk_preds)
        assert r2 >= (sk_r2 - 0.09)


def check_predict_proba(test_proba, baseline_proba, y_test, rel_err):
    y_proba = np.zeros(np.shape(baseline_proba))
    for count, _class in enumerate(y_test):
        y_proba[count, _class] = 1
    baseline_mse = mean_squared_error(y_proba, baseline_proba)
    test_mse = mean_squared_error(y_proba, test_proba)
    # using relative error is more stable when changing decision tree
    # parameters, column or class count
    assert test_mse <= baseline_mse * (1.0 + rel_err)


def rf_classification(
    datatype, array_type, max_features, max_samples, fixture
):
    X, y = fixture
    X = X.astype(datatype[0])
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )
    X_test = X_test.astype(datatype[1])

    n_streams = 1
    handle, stream = get_handle(True, n_streams=n_streams)
    # Initialize, fit and predict using cuML's
    # random forest classification model
    cuml_model = curfc(
        max_features=max_features,
        max_samples=max_samples,
        n_bins=16,
        split_criterion=0,
        min_samples_leaf=2,
        random_state=999,
        n_estimators=40,
        handle=handle,
        max_leaves=-1,
        max_depth=16,
        n_streams=n_streams,
    )
    if array_type == "dataframe":
        X_train_df = cudf.DataFrame(X_train)
        y_train_df = cudf.Series(y_train)
        X_test_df = cudf.DataFrame(X_test)
        cuml_model.fit(X_train_df, y_train_df)
        cu_proba_gpu = cuml_model.predict_proba(X_test_df).to_numpy()
        cu_preds_gpu = cuml_model.predict(X_test_df).to_numpy()
    else:
        cuml_model.fit(X_train, y_train)
        cu_proba_gpu = cuml_model.predict_proba(X_test)
        cu_preds_gpu = cuml_model.predict(X_test)
    np.testing.assert_array_equal(
        cu_preds_gpu, np.argmax(cu_proba_gpu, axis=1)
    )
    cu_acc_gpu = accuracy_score(y_test, cu_preds_gpu)

    # sklearn random forest classification model
    # initialization, fit and predict
    if y.size < 500000:
        sk_model = skrfc(
            n_estimators=40,
            max_depth=16,
            min_samples_split=2,
            max_features=max_features,
            random_state=10,
        )
        sk_model.fit(X_train, y_train)
        sk_preds = sk_model.predict(X_test)
        sk_acc = accuracy_score(y_test, sk_preds)
        sk_proba = sk_model.predict_proba(X_test)
        assert cu_acc_gpu >= sk_acc - 0.07
        # 0.06 is the highest relative error observed on CI, within
        # 0.0061 absolute error boundaries seen previously
        check_predict_proba(cu_proba_gpu, sk_proba, y_test, 0.1)


@pytest.mark.parametrize("datatype", [(np.float32, np.float64)])
@pytest.mark.parametrize("array_type", ["dataframe", "numpy"])
@pytest.mark.skipif(
    cudf_pandas_active,
    reason="cudf.pandas causes sklearn RF estimators crashes sometimes. "
    "Issue: https://github.com/rapidsai/cuml/issues/5991",
)
def test_rf_classification_multi_class(mclass_clf, datatype, array_type):
    rf_classification(datatype, array_type, 1.0, 1.0, mclass_clf)


@pytest.mark.parametrize("datatype", [(np.float32, np.float64)])
@pytest.mark.parametrize("max_samples", [unit_param(1.0), stress_param(0.95)])
@pytest.mark.parametrize("max_features", [1.0, "log2", "sqrt"])
@pytest.mark.skipif(
    cudf_pandas_active,
    reason="cudf.pandas causes sklearn RF estimators crashes sometimes. "
    "Issue: https://github.com/rapidsai/cuml/issues/5991",
)
def test_rf_classification_proba(
    small_clf, datatype, max_samples, max_features
):
    rf_classification(datatype, "numpy", max_features, max_samples, small_clf)


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "fil_layout", ["depth_first", "breadth_first", "layered"]
)
@pytest.mark.skipif(
    cudf_pandas_active,
    reason="cudf.pandas causes sklearn RF estimators crashes sometimes. "
    "Issue: https://github.com/rapidsai/cuml/issues/5991",
)
def test_rf_classification_sparse(small_clf, datatype, fil_layout):
    use_handle = True
    num_trees = 50

    X, y = small_clf
    X = X.astype(datatype)
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )

    # Create a handle for the cuml model
    handle, stream = get_handle(use_handle, n_streams=1)

    # Initialize, fit and predict using cuML's
    # random forest classification model
    cuml_model = curfc(
        n_bins=16,
        split_criterion=0,
        min_samples_leaf=2,
        random_state=123,
        n_streams=1,
        n_estimators=num_trees,
        handle=handle,
        max_leaves=-1,
        max_depth=40,
    )
    cuml_model.fit(X_train, y_train)
    preds = cuml_model.predict(X_test, layout=fil_layout)
    acc = accuracy_score(y_test, preds)
    np.testing.assert_almost_equal(acc, cuml_model.score(X_test, y_test))

    fil_model = cuml_model.as_fil()

    with cuml.using_output_type("numpy"):
        fil_model_preds = fil_model.predict(X_test)
        fil_model_acc = accuracy_score(y_test, fil_model_preds)
        assert acc == fil_model_acc

    tl_model = cuml_model.as_treelite()
    assert num_trees == tl_model.num_tree
    assert X.shape[1] == tl_model.num_feature

    if X.shape[0] < 500000:
        sk_model = skrfc(
            n_estimators=50,
            max_depth=40,
            min_samples_split=2,
            random_state=10,
        )
        sk_model.fit(X_train, y_train)
        sk_preds = sk_model.predict(X_test)
        sk_acc = accuracy_score(y_test, sk_preds)
        assert acc >= (sk_acc - 0.07)


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "fil_layout", ["depth_first", "breadth_first", "layered"]
)
@pytest.mark.skipif(
    cudf_pandas_active,
    reason="cudf.pandas causes sklearn RF estimators crashes sometimes. "
    "Issue: https://github.com/rapidsai/cuml/issues/5991",
)
def test_rf_regression_sparse(special_reg, datatype, fil_layout):
    use_handle = True
    num_trees = 50

    X, y = special_reg
    X = X.astype(datatype)
    y = y.astype(datatype)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )

    # Create a handle for the cuml model
    handle, stream = get_handle(use_handle, n_streams=1)

    # Initialize and fit using cuML's random forest regression model
    cuml_model = curfr(
        n_bins=16,
        split_criterion=2,
        min_samples_leaf=2,
        random_state=123,
        n_streams=1,
        n_estimators=num_trees,
        handle=handle,
        max_leaves=-1,
        max_depth=40,
    )
    cuml_model.fit(X_train, y_train)

    preds = cuml_model.predict(X_test, layout=fil_layout)
    r2 = r2_score(y_test, preds)

    fil_model = cuml_model.as_fil()

    with cuml.using_output_type("numpy"):
        fil_model_preds = fil_model.predict(X_test)
        fil_model_preds = np.reshape(fil_model_preds, np.shape(y_test))
        fil_model_r2 = r2_score(y_test, fil_model_preds)
        assert r2 == fil_model_r2

    tl_model = cuml_model.as_treelite()
    assert num_trees == tl_model.num_tree
    assert X.shape[1] == tl_model.num_feature

    # Initialize, fit and predict using
    # sklearn's random forest regression model
    if X.shape[0] < 1000:  # mode != "stress":
        sk_model = skrfr(
            n_estimators=50,
            max_depth=40,
            min_samples_split=2,
            random_state=10,
        )
        sk_model.fit(X_train, y_train)
        sk_preds = sk_model.predict(X_test)
        sk_r2 = r2_score(y_test, sk_preds)
        assert r2 >= (sk_r2 - 0.08)


@pytest.mark.xfail(reason="Need rapidsai/rmm#415 to detect memleak robustly")
@pytest.mark.memleak
@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "fil_layout", ["depth_first", "breadth_first", "layered"]
)
@pytest.mark.parametrize(
    "n_iter", [unit_param(5), quality_param(30), stress_param(80)]
)
def test_rf_memory_leakage(small_clf, datatype, fil_layout, n_iter):
    use_handle = True

    X, y = small_clf
    X = X.astype(datatype)
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )

    # Create a handle for the cuml model
    handle, stream = get_handle(use_handle, n_streams=1)

    # Warmup. Some modules that are used in RF allocate space on the device
    # and consume memory. This is to make sure that the allocation is done
    # before the first call to get_memory_info.
    base_model = curfc(handle=handle)
    base_model.fit(X_train, y_train)
    handle.sync()  # just to be sure
    free_mem = cuda.current_context().get_memory_info()[0]

    def test_for_memory_leak():
        nonlocal free_mem
        cuml_mods = curfc(handle=handle)
        cuml_mods.fit(X_train, y_train)
        handle.sync()  # just to be sure
        # Calculate the memory free after fitting the cuML model
        delta_mem = free_mem - cuda.current_context().get_memory_info()[0]
        assert delta_mem == 0

        for i in range(2):
            cuml_mods.predict(X_test, layout=fil_layout)
            handle.sync()  # just to be sure
            # Calculate the memory free after predicting the cuML model
            delta_mem = free_mem - cuda.current_context().get_memory_info()[0]
            assert delta_mem == 0

    for i in range(n_iter):
        test_for_memory_leak()


@pytest.mark.parametrize("max_features", [1.0, "log2", "sqrt"])
@pytest.mark.parametrize("max_depth", [10, 13, 16])
@pytest.mark.parametrize("n_estimators", [10, 20, 100])
@pytest.mark.parametrize("n_bins", [8, 9, 10])
def test_create_classification_model(
    max_features, max_depth, n_estimators, n_bins
):

    # random forest classification model
    cuml_model = curfc(
        max_features=max_features,
        n_bins=n_bins,
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    params = cuml_model.get_params()
    cuml_model2 = curfc()
    cuml_model2.set_params(**params)
    verfiy_params = cuml_model2.get_params()
    assert params["max_features"] == verfiy_params["max_features"]
    assert params["max_depth"] == verfiy_params["max_depth"]
    assert params["n_estimators"] == verfiy_params["n_estimators"]
    assert params["n_bins"] == verfiy_params["n_bins"]


@pytest.mark.parametrize("n_estimators", [10, 20, 100])
@pytest.mark.parametrize("n_bins", [8, 9, 10])
def test_multiple_fits_classification(large_clf, n_estimators, n_bins):

    datatype = np.float32
    X, y = large_clf
    X = X.astype(datatype)
    y = y.astype(np.int32)
    cuml_model = curfc(n_bins=n_bins, n_estimators=n_estimators, max_depth=10)

    # Calling multiple fits
    cuml_model.fit(X, y)

    cuml_model.fit(X, y)

    # Check if params are still intact
    params = cuml_model.get_params()
    assert params["n_estimators"] == n_estimators
    assert params["n_bins"] == n_bins


@pytest.mark.parametrize(
    "column_info",
    [
        unit_param([100, 50]),
        quality_param([200, 100]),
        stress_param([500, 350]),
    ],
)
@pytest.mark.parametrize(
    "nrows", [unit_param(500), quality_param(5000), stress_param(500000)]
)
@pytest.mark.parametrize("n_estimators", [10, 20, 100])
@pytest.mark.parametrize("n_bins", [8, 9, 10])
def test_multiple_fits_regression(column_info, nrows, n_estimators, n_bins):
    datatype = np.float32
    ncols, n_info = column_info
    X, y = make_regression(
        n_samples=nrows,
        n_features=ncols,
        n_informative=n_info,
        random_state=123,
    )
    X = X.astype(datatype)
    y = y.astype(np.int32)
    cuml_model = curfr(n_bins=n_bins, n_estimators=n_estimators, max_depth=10)

    # Calling multiple fits
    cuml_model.fit(X, y)

    cuml_model.fit(X, y)

    cuml_model.fit(X, y)

    # Check if params are still intact
    params = cuml_model.get_params()
    assert params["n_estimators"] == n_estimators
    assert params["n_bins"] == n_bins


@pytest.mark.xfail(
    reason="Needs refactoring/debugging due to sporadic failures"
    "https://github.com/rapidsai/cuml/issues/5528"
)
@pytest.mark.memleak
@pytest.mark.parametrize("estimator_type", ["classification"])
def test_rf_host_memory_leak(large_clf, estimator_type):
    import gc
    import os

    try:
        import psutil
    except ImportError:
        pytest.skip("psutil not installed")

    process = psutil.Process(os.getpid())

    X, y = large_clf
    X = X.astype(np.float32)
    params = {"max_depth": 50}
    if estimator_type == "classification":
        base_model = curfc(max_depth=10, n_estimators=100, random_state=123)
        y = y.astype(np.int32)
    else:
        base_model = curfr(max_depth=10, n_estimators=100, random_state=123)
        y = y.astype(np.float32)

    # Pre-fit once - this is our baseline and memory usage
    # should not significantly exceed it after later fits
    base_model.fit(X, y)
    gc.collect()
    initial_baseline_mem = process.memory_info().rss

    for i in range(5):
        base_model.fit(X, y)
        base_model.set_params(**params)
        gc.collect()
        final_mem = process.memory_info().rss

    # Some tiny allocations may occur, but we should not leak
    # without bounds, which previously happened
    assert (final_mem - initial_baseline_mem) < 2.4e6


@pytest.mark.xfail(
    reason="Needs refactoring/debugging due to sporadic failures"
    "https://github.com/rapidsai/cuml/issues/5528"
)
@pytest.mark.memleak
@pytest.mark.parametrize("estimator_type", ["regression", "classification"])
@pytest.mark.parametrize("i", list(range(100)))
def test_concat_memory_leak(large_clf, estimator_type, i):
    import gc
    import os

    try:
        import psutil
    except ImportError:
        pytest.skip("psutil not installed")

    process = psutil.Process(os.getpid())

    X, y = large_clf
    X = X.astype(np.float32)

    # Build a series of RF models
    n_models = 10
    if estimator_type == "classification":
        base_models = [
            curfc(max_depth=10, n_estimators=100, random_state=123)
            for i in range(n_models)
        ]
        y = y.astype(np.int32)
    elif estimator_type == "regression":
        base_models = [
            curfr(max_depth=10, n_estimators=100, random_state=123)
            for i in range(n_models)
        ]
        y = y.astype(np.float32)
    else:
        assert False

    # Pre-fit once - this is our baseline and memory usage
    # should not significantly exceed it after later fits
    for model in base_models:
        model.fit(X, y)

    # Just concatenate over and over in a loop
    concat_models = base_models[1:]
    init_model = base_models[0]
    other_handles = [
        model._obtain_treelite_handle() for model in concat_models
    ]
    init_model._concatenate_treelite_handle(other_handles)

    gc.collect()
    initial_baseline_mem = process.memory_info().rss
    for i in range(10):
        init_model._concatenate_treelite_handle(other_handles)
        gc.collect()
        used_mem = process.memory_info().rss
        logger.debug(
            "memory at rep %2d: %d m"
            % (i, (used_mem - initial_baseline_mem) / 1e6)
        )

    gc.collect()
    used_mem = process.memory_info().rss
    logger.info(
        "Final memory delta: %d" % ((used_mem - initial_baseline_mem) / 1e6)
    )

    # increasing margin to avoid very infrequent failures
    assert (used_mem - initial_baseline_mem) < 1.1e6


def test_rf_nbins_small(small_clf):
    X, y = small_clf
    X = X.astype(np.float32)
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )
    # Initialize, fit and predict using cuML's
    # random forest classification model
    cuml_model = curfc()

    # display warning when nbins less than samples
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cuml_model.fit(X_train[0:3, :], y_train[0:3])
        assert (
            "The number of bins, `n_bins` is greater than "
            "the number of samples used for training. "
            "Changing `n_bins` to number of training samples."
            in str(w[-1].message)
        )


def test_rf_regression_with_identical_labels():
    X = np.array([[-1, 0], [0, 1], [2, 0], [0, 3], [-2, 0]], dtype=np.float32)
    y = np.array([1, 1, 1, 1, 1], dtype=np.float32)
    # Degenerate case: all labels are identical.
    # RF Regressor must not create any split. It must yield an empty tree
    # with only the root node.
    model = curfr(
        max_features=1.0,
        max_samples=1.0,
        n_bins=5,
        bootstrap=False,
        split_criterion="mse",
        min_samples_leaf=1,
        min_samples_split=2,
        random_state=0,
        n_streams=1,
        n_estimators=1,
        max_depth=1,
    )
    model.fit(X, y)
    trees = json.loads(model.as_treelite().dump_as_json())["trees"]
    assert len(trees) == 1
    assert len(trees[0]["nodes"]) == 1
    assert trees[0]["nodes"][0] == {
        "node_id": 0,
        "leaf_value": 1.0,
        "data_count": 5,
    }


def test_rf_regressor_gtil_integration(tmpdir):
    X, y = fetch_california_housing(return_X_y=True)
    X, y = X.astype(np.float32), y.astype(np.float32)
    clf = curfr(max_depth=3, random_state=0, n_estimators=10)
    clf.fit(X, y)
    expected_pred = clf.predict(X).reshape((-1, 1, 1))

    checkpoint_path = os.path.join(tmpdir, "checkpoint.tl")
    clf.as_treelite().serialize(checkpoint_path)

    tl_model = treelite.Model.deserialize(checkpoint_path)
    out_pred = treelite.gtil.predict(tl_model, X)
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)


def test_rf_binary_classifier_gtil_integration(tmpdir):
    X, y = load_breast_cancer(return_X_y=True)
    X, y = X.astype(np.float32), y.astype(np.int32)
    clf = curfc(max_depth=3, random_state=0, n_estimators=10)
    clf.fit(X, y)
    expected_pred = clf.predict_proba(X).reshape((-1, 1, 2))

    checkpoint_path = os.path.join(tmpdir, "checkpoint.tl")
    clf.as_treelite().serialize(checkpoint_path)

    tl_model = treelite.Model.deserialize(checkpoint_path)
    out_pred = treelite.gtil.predict(tl_model, X)
    np.testing.assert_almost_equal(out_pred, expected_pred, decimal=5)


def test_rf_multiclass_classifier_gtil_integration(tmpdir):
    X, y = load_iris(return_X_y=True)
    X, y = X.astype(np.float32), y.astype(np.int32)
    clf = curfc(max_depth=3, random_state=0, n_estimators=10)
    clf.fit(X, y)
    expected_prob = clf.predict_proba(X).reshape((X.shape[0], 1, -1))

    checkpoint_path = os.path.join(tmpdir, "checkpoint.tl")
    clf.as_treelite().serialize(checkpoint_path)

    tl_model = treelite.Model.deserialize(checkpoint_path)
    out_prob = treelite.gtil.predict(tl_model, X, pred_margin=True)
    np.testing.assert_almost_equal(out_prob, expected_prob, decimal=5)


@pytest.mark.parametrize(
    "estimator, make_data",
    [
        (curfc, make_classification),
        (curfr, make_regression),
    ],
)
def test_rf_min_samples_split_with_small_float(estimator, make_data):
    # Check that min_samples leaf is works with a small float
    # Non-regression test for gh-4613
    X, y = make_data(random_state=0)
    clf = estimator(min_samples_split=0.0001, random_state=0, n_estimators=2)

    # Capture and verify expected warning
    warning_msg = (
        "The number of bins, `n_bins` is greater than the number of samples "
        "used for training"
    )
    with pytest.warns(UserWarning, match=warning_msg):
        clf.fit(X, y)


@pytest.mark.parametrize(
    "max_features, sol",
    [
        (2, 0.02),
        (0.5, 0.5),
        ("sqrt", math.sqrt(100) / 100),
        ("log2", math.log2(100) / 100),
        (None, 1.0),
    ],
)
def test_max_features(max_features, sol):
    res = compute_max_features(max_features, 100)
    assert res == sol


def test_rf_predict_returns_int():

    X, y = make_classification()

    # Capture and verify expected warning
    warning_msg = (
        "The number of bins, `n_bins` is greater than the number of samples "
        "used for training"
    )
    with pytest.warns(UserWarning, match=warning_msg):
        clf = cuml.ensemble.RandomForestClassifier().fit(X, y)

    pred = clf.predict(X)
    assert pred.dtype == np.int64


def test_ensemble_estimator_length():
    X, y = make_classification()
    clf = cuml.ensemble.RandomForestClassifier(n_estimators=3)

    # Capture and verify expected warning
    warning_msg = (
        "The number of bins, `n_bins` is greater than the number of samples "
        "used for training"
    )
    with pytest.warns(UserWarning, match=warning_msg):
        clf.fit(X, y)

    assert len(clf) == 3

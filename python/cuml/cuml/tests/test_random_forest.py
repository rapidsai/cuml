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

import treelite
from sklearn.model_selection import train_test_split
from sklearn.datasets import (
    fetch_california_housing,
    make_classification,
    make_regression,
    load_iris,
    load_breast_cancer,
)
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    mean_tweedie_deviance,
)
from sklearn.ensemble import RandomForestRegressor as skrfr
from sklearn.ensemble import RandomForestClassifier as skrfc
import cuml.internals.logger as logger
from cuml.testing.utils import (
    get_handle,
    unit_param,
    quality_param,
    stress_param,
)
from cuml.metrics import r2_score
from cuml.ensemble import RandomForestRegressor as curfr
from cuml.ensemble import RandomForestClassifier as curfc
import cuml
from cuml.internals.safe_imports import gpu_only_import_from
import os
import json
import random
from cuml.internals.safe_imports import cpu_only_import
import pytest

import warnings
from cuml.internals.safe_imports import gpu_only_import

cudf = gpu_only_import("cudf")
np = cpu_only_import("numpy")

cuda = gpu_only_import_from("numba", "cuda")
cudf_pandas_active = gpu_only_import_from("cudf.pandas", "LOADED")


pytestmark = pytest.mark.filterwarnings(
    "ignore: For reproducible results(.*)" "::cuml[.*]"
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
    )
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

    fil_preds = cuml_model.predict(
        X_test, predict_model="GPU", threshold=0.5, algo="auto"
    )
    cu_preds = cuml_model.predict(X_test, predict_model="CPU")
    fil_preds = np.reshape(fil_preds, np.shape(cu_preds))
    cuml_acc = accuracy_score(y_test, cu_preds)
    fil_acc = accuracy_score(y_test, fil_preds)
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
        assert fil_acc >= (sk_acc - 0.07)
    assert fil_acc >= (
        cuml_acc - 0.07
    )  # to be changed to 0.02. see issue #3910: https://github.com/rapidsai/cuml/issues/3910 # noqa


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

    fil_preds = cuml_model.predict(
        X_test, predict_model="GPU", threshold=0.5, algo="auto"
    )
    cu_preds = cuml_model.predict(X_test, predict_model="CPU")
    fil_preds = np.reshape(fil_preds, np.shape(cu_preds))
    cuml_acc = accuracy_score(y_test, cu_preds)
    fil_acc = accuracy_score(y_test, fil_preds)
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
        assert fil_acc >= (sk_acc - 0.07)
    assert fil_acc >= (
        cuml_acc - 0.07
    )  # to be changed to 0.02. see issue #3910: https://github.com/rapidsai/cuml/issues/3910 # noqa


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
        accuracy_metric="mse",
    )
    cuml_model.fit(X_train, y_train)
    # predict using FIL
    fil_preds = cuml_model.predict(X_test, predict_model="GPU")
    cu_preds = cuml_model.predict(X_test, predict_model="CPU")
    fil_preds = np.reshape(fil_preds, np.shape(cu_preds))

    cu_r2 = r2_score(y_test, cu_preds)
    fil_r2 = r2_score(y_test, fil_preds)
    # Initialize, fit and predict using
    # sklearn's random forest regression model
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
        assert fil_r2 >= (sk_r2 - 0.07)
    assert fil_r2 >= (cu_r2 - 0.02)


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
        # Initialize, fit and predict using cuML's
        # random forest classification model
        cu_class = curfc(random_state=seed, n_streams=1)
        cu_class.fit(X_train, y_train)

        # predict using FIL
        fil_preds_orig = cu_class.predict(X_test, predict_model="GPU")
        cu_preds_orig = cu_class.predict(X_test, predict_model="CPU")
        cu_acc_orig = accuracy_score(y_test, cu_preds_orig)
        fil_preds_orig = np.reshape(fil_preds_orig, np.shape(cu_preds_orig))

        fil_acc_orig = accuracy_score(y_test, fil_preds_orig)

        # Initialize, fit and predict using cuML's
        # random forest classification model
        cu_class2 = curfc(random_state=seed, n_streams=1)
        cu_class2.fit(X_train, y_train)

        # predict using FIL
        fil_preds_rerun = cu_class2.predict(X_test, predict_model="GPU")
        cu_preds_rerun = cu_class2.predict(X_test, predict_model="CPU")
        cu_acc_rerun = accuracy_score(y_test, cu_preds_rerun)
        fil_preds_rerun = np.reshape(fil_preds_rerun, np.shape(cu_preds_rerun))

        fil_acc_rerun = accuracy_score(y_test, fil_preds_rerun)

        assert fil_acc_orig == fil_acc_rerun
        assert cu_acc_orig == cu_acc_rerun
        assert (fil_preds_orig == fil_preds_rerun).all()
        assert (cu_preds_orig == cu_preds_rerun).all()


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
def test_rf_classification_float64(small_clf, datatype, convert_dtype):

    X, y = small_clf
    X = X.astype(datatype[0])
    y = y.astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )
    X_test = X_test.astype(datatype[1])

    # Initialize, fit and predict using cuML's
    # random forest classification model
    cuml_model = curfc()
    cuml_model.fit(X_train, y_train)
    cu_preds = cuml_model.predict(X_test, predict_model="CPU")
    cu_acc = accuracy_score(y_test, cu_preds)

    # sklearn random forest classification model
    # initialization, fit and predict
    if X.shape[0] < 500000:
        sk_model = skrfc(max_depth=16, random_state=10)
        sk_model.fit(X_train, y_train)
        sk_preds = sk_model.predict(X_test)
        sk_acc = accuracy_score(y_test, sk_preds)
        assert cu_acc >= (sk_acc - 0.07)

    # predict using cuML's GPU based prediction
    fil_preds = cuml_model.predict(
        X_test, predict_model="GPU", convert_dtype=convert_dtype
    )
    fil_preds = np.reshape(fil_preds, np.shape(cu_preds))

    fil_acc = accuracy_score(y_test, fil_preds)
    assert fil_acc >= (
        cu_acc - 0.07
    )  # to be changed to 0.02. see issue #3910: https://github.com/rapidsai/cuml/issues/3910 # noqa


@pytest.mark.parametrize(
    "datatype", [(np.float64, np.float32), (np.float32, np.float64)]
)
@pytest.mark.filterwarnings("ignore:To use pickling(.*)::cuml[.*]")
@pytest.mark.skipif(
    cudf_pandas_active,
    reason="cudf.pandas causes sklearn RF estimators crashes sometimes. "
    "Issue: https://github.com/rapidsai/cuml/issues/5991",
)
def test_rf_regression_float64(large_reg, datatype):

    X, y = large_reg
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=0
    )
    X_train = X_train.astype(datatype[0])
    y_train = y_train.astype(datatype[0])
    X_test = X_test.astype(datatype[1])
    y_test = y_test.astype(datatype[1])

    # Initialize, fit and predict using cuML's
    # random forest classification model
    cuml_model = curfr()
    cuml_model.fit(X_train, y_train)
    cu_preds = cuml_model.predict(X_test, predict_model="CPU")
    cu_r2 = r2_score(y_test, cu_preds)

    # sklearn random forest classification model
    # initialization, fit and predict
    if X.shape[0] < 500000:
        sk_model = skrfr(max_depth=16, random_state=10)
        sk_model.fit(X_train, y_train)
        sk_preds = sk_model.predict(X_test)
        sk_r2 = r2_score(y_test, sk_preds)
        assert cu_r2 >= (sk_r2 - 0.09)

    # predict using cuML's GPU based prediction
    fil_preds = cuml_model.predict(
        X_test, predict_model="GPU", convert_dtype=True
    )
    fil_preds = np.reshape(fil_preds, np.shape(cu_preds))
    fil_r2 = r2_score(y_test, fil_preds)
    assert fil_r2 >= (cu_r2 - 0.02)


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
        cu_preds_cpu = cuml_model.predict(
            X_test_df, predict_model="CPU"
        ).to_numpy()
        cu_preds_gpu = cuml_model.predict(
            X_test_df, predict_model="GPU"
        ).to_numpy()
    else:
        cuml_model.fit(X_train, y_train)
        cu_proba_gpu = cuml_model.predict_proba(X_test)
        cu_preds_cpu = cuml_model.predict(X_test, predict_model="CPU")
        cu_preds_gpu = cuml_model.predict(X_test, predict_model="GPU")
    np.testing.assert_array_equal(
        cu_preds_gpu, np.argmax(cu_proba_gpu, axis=1)
    )

    cu_acc_cpu = accuracy_score(y_test, cu_preds_cpu)
    cu_acc_gpu = accuracy_score(y_test, cu_preds_gpu)
    assert cu_acc_cpu == pytest.approx(cu_acc_gpu, abs=0.01, rel=0.1)

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
        assert cu_acc_cpu >= sk_acc - 0.07
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
    "fil_sparse_format", ["not_supported", True, "auto", False]
)
@pytest.mark.parametrize(
    "algo", ["auto", "naive", "tree_reorg", "batch_tree_reorg"]
)
@pytest.mark.skipif(
    cudf_pandas_active,
    reason="cudf.pandas causes sklearn RF estimators crashes sometimes. "
    "Issue: https://github.com/rapidsai/cuml/issues/5991",
)
def test_rf_classification_sparse(
    small_clf, datatype, fil_sparse_format, algo
):
    use_handle = True
    num_treees = 50

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
        n_estimators=num_treees,
        handle=handle,
        max_leaves=-1,
        max_depth=40,
    )
    cuml_model.fit(X_train, y_train)

    if (
        not fil_sparse_format
        or algo == "tree_reorg"
        or algo == "batch_tree_reorg"
    ) or fil_sparse_format == "not_supported":
        with pytest.raises(ValueError):
            fil_preds = cuml_model.predict(
                X_test,
                predict_model="GPU",
                threshold=0.5,
                fil_sparse_format=fil_sparse_format,
                algo=algo,
            )
    else:
        fil_preds = cuml_model.predict(
            X_test,
            predict_model="GPU",
            threshold=0.5,
            fil_sparse_format=fil_sparse_format,
            algo=algo,
        )
        fil_preds = np.reshape(fil_preds, np.shape(y_test))
        fil_acc = accuracy_score(y_test, fil_preds)
        np.testing.assert_almost_equal(
            fil_acc, cuml_model.score(X_test, y_test)
        )

        fil_model = cuml_model.convert_to_fil_model()

        with cuml.using_output_type("numpy"):
            fil_model_preds = fil_model.predict(X_test)
            fil_model_acc = accuracy_score(y_test, fil_model_preds)
            assert fil_acc == fil_model_acc

        tl_model = cuml_model.convert_to_treelite_model()
        assert num_treees == tl_model.num_trees
        assert X.shape[1] == tl_model.num_features

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
            assert fil_acc >= (sk_acc - 0.07)


@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "fil_sparse_format", ["not_supported", True, "auto", False]
)
@pytest.mark.parametrize(
    "algo", ["auto", "naive", "tree_reorg", "batch_tree_reorg"]
)
@pytest.mark.skipif(
    cudf_pandas_active,
    reason="cudf.pandas causes sklearn RF estimators crashes sometimes. "
    "Issue: https://github.com/rapidsai/cuml/issues/5991",
)
def test_rf_regression_sparse(special_reg, datatype, fil_sparse_format, algo):
    use_handle = True
    num_treees = 50

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
        n_estimators=num_treees,
        handle=handle,
        max_leaves=-1,
        max_depth=40,
        accuracy_metric="mse",
    )
    cuml_model.fit(X_train, y_train)

    # predict using FIL
    if (
        not fil_sparse_format
        or algo == "tree_reorg"
        or algo == "batch_tree_reorg"
    ) or fil_sparse_format == "not_supported":
        with pytest.raises(ValueError):
            fil_preds = cuml_model.predict(
                X_test,
                predict_model="GPU",
                fil_sparse_format=fil_sparse_format,
                algo=algo,
            )
    else:
        fil_preds = cuml_model.predict(
            X_test,
            predict_model="GPU",
            fil_sparse_format=fil_sparse_format,
            algo=algo,
        )
        fil_preds = np.reshape(fil_preds, np.shape(y_test))
        fil_r2 = r2_score(y_test, fil_preds)

        fil_model = cuml_model.convert_to_fil_model()

        with cuml.using_output_type("numpy"):
            fil_model_preds = fil_model.predict(X_test)
            fil_model_preds = np.reshape(fil_model_preds, np.shape(y_test))
            fil_model_r2 = r2_score(y_test, fil_model_preds)
            assert fil_r2 == fil_model_r2

        tl_model = cuml_model.convert_to_treelite_model()
        assert num_treees == tl_model.num_trees
        assert X.shape[1] == tl_model.num_features

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
            assert fil_r2 >= (sk_r2 - 0.08)


@pytest.mark.xfail(reason="Need rapidsai/rmm#415 to detect memleak robustly")
@pytest.mark.memleak
@pytest.mark.parametrize("datatype", [np.float32, np.float64])
@pytest.mark.parametrize("fil_sparse_format", [True, False, "auto"])
@pytest.mark.parametrize(
    "n_iter", [unit_param(5), quality_param(30), stress_param(80)]
)
def test_rf_memory_leakage(small_clf, datatype, fil_sparse_format, n_iter):
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
        cuml_mods = curfc(handle=handle)
        cuml_mods.fit(X_train, y_train)
        handle.sync()  # just to be sure
        # Calculate the memory free after fitting the cuML model
        delta_mem = free_mem - cuda.current_context().get_memory_info()[0]
        assert delta_mem == 0

        for i in range(2):
            cuml_mods.predict(
                X_test,
                predict_model="GPU",
                fil_sparse_format=fil_sparse_format,
            )
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


@pytest.mark.parametrize("n_estimators", [5, 10, 20])
@pytest.mark.parametrize("detailed_text", [True, False])
def test_rf_get_text(n_estimators, detailed_text):

    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_clusters_per_class=1,
        n_informative=5,
        random_state=94929,
        n_classes=2,
    )

    X = X.astype(np.float32)
    y = y.astype(np.int32)

    # Create a handle for the cuml model
    handle, stream = get_handle(True, n_streams=1)

    # Initialize cuML Random Forest classification model
    cuml_model = curfc(
        handle=handle,
        max_features=1.0,
        max_samples=1.0,
        n_bins=16,
        split_criterion=0,
        min_samples_leaf=2,
        random_state=23707,
        n_streams=1,
        n_estimators=n_estimators,
        max_leaves=-1,
        max_depth=16,
    )

    # Train model on the data
    cuml_model.fit(X, y)

    if detailed_text:
        text_output = cuml_model.get_detailed_text()
    else:
        text_output = cuml_model.get_summary_text()

    # Test 1: Output is non-zero
    assert "" != text_output

    # Count the number of trees printed
    tree_count = 0
    for line in text_output.split("\n"):
        if line.strip().startswith("Tree #"):
            tree_count += 1

    # Test 2: Correct number of trees are printed
    assert n_estimators == tree_count


@pytest.mark.parametrize("max_depth", [1, 2, 3, 5, 10, 15, 20])
@pytest.mark.parametrize("n_estimators", [5, 10, 20])
@pytest.mark.parametrize("estimator_type", ["regression", "classification"])
def test_rf_get_json(estimator_type, max_depth, n_estimators):
    X, y = make_classification(
        n_samples=350,
        n_features=20,
        n_clusters_per_class=1,
        n_informative=10,
        random_state=123,
        n_classes=2,
    )
    X = X.astype(np.float32)
    if estimator_type == "classification":
        cuml_model = curfc(
            max_features=1.0,
            max_samples=1.0,
            n_bins=16,
            split_criterion=0,
            min_samples_leaf=2,
            random_state=23707,
            n_streams=1,
            n_estimators=n_estimators,
            max_leaves=-1,
            max_depth=max_depth,
        )
        y = y.astype(np.int32)
    elif estimator_type == "regression":
        cuml_model = curfr(
            max_features=1.0,
            max_samples=1.0,
            n_bins=16,
            min_samples_leaf=2,
            random_state=23707,
            n_streams=1,
            n_estimators=n_estimators,
            max_leaves=-1,
            max_depth=max_depth,
        )
        y = y.astype(np.float32)
    else:
        assert False

    # Train model on the data
    cuml_model.fit(X, y)

    json_out = cuml_model.get_json()
    json_obj = json.loads(json_out)

    # Test 1: Output is non-zero
    assert "" != json_out

    # Test 2: JSON object contains correct number of trees
    assert isinstance(json_obj, list)
    assert len(json_obj) == n_estimators

    # Test 3: Traverse JSON trees and get the same predictions as cuML RF
    def predict_with_json_tree(tree, x):
        if "children" not in tree:
            assert "leaf_value" in tree
            return tree["leaf_value"]
        assert "split_feature" in tree
        assert "split_threshold" in tree
        assert "yes" in tree
        assert "no" in tree
        if x[tree["split_feature"]] <= tree["split_threshold"] + 1e-5:
            return predict_with_json_tree(tree["children"][0], x)
        return predict_with_json_tree(tree["children"][1], x)

    def predict_with_json_rf_classifier(rf, x):
        # Returns the class with the highest vote. If there is a tie, return
        # the list of all classes with the highest vote.
        predictions = []
        for tree in rf:
            predictions.append(np.array(predict_with_json_tree(tree, x)))
        predictions = np.sum(predictions, axis=0)
        return np.argmax(predictions)

    def predict_with_json_rf_regressor(rf, x):
        pred = 0.0
        for tree in rf:
            pred += predict_with_json_tree(tree, x)[0]
        return pred / len(rf)

    if estimator_type == "classification":
        expected_pred = cuml_model.predict(X).astype(np.int32)
        for idx, row in enumerate(X):
            majority_vote = predict_with_json_rf_classifier(json_obj, row)
            assert expected_pred[idx] == majority_vote
    elif estimator_type == "regression":
        expected_pred = cuml_model.predict(X).astype(np.float32)
        pred = []
        for idx, row in enumerate(X):
            pred.append(predict_with_json_rf_regressor(json_obj, row))
        pred = np.array(pred, dtype=np.float32)
        print(json_obj)
        for i in range(len(pred)):
            assert np.isclose(pred[i], expected_pred[i]), X[i, 19]
        np.testing.assert_almost_equal(pred, expected_pred, decimal=6)


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


@pytest.mark.parametrize("split_criterion", [2], ids=["mse"])
def test_rf_regression_with_identical_labels(split_criterion):
    X = np.array([[-1, 0], [0, 1], [2, 0], [0, 3], [-2, 0]], dtype=np.float32)
    y = np.array([1, 1, 1, 1, 1], dtype=np.float32)
    # Degenerate case: all labels are identical.
    # RF Regressor must not create any split. It must yield an empty tree
    # with only the root node.
    clf = curfr(
        max_features=1.0,
        max_samples=1.0,
        n_bins=5,
        bootstrap=False,
        split_criterion=split_criterion,
        min_samples_leaf=1,
        min_samples_split=2,
        random_state=0,
        n_streams=1,
        n_estimators=1,
        max_depth=1,
    )
    clf.fit(X, y)
    model_dump = json.loads(clf.get_json())
    assert len(model_dump) == 1
    expected_dump = {"nodeid": 0, "leaf_value": [1.0], "instance_count": 5}
    assert model_dump[0] == expected_dump


def test_rf_regressor_gtil_integration(tmpdir):
    X, y = fetch_california_housing(return_X_y=True)
    X, y = X.astype(np.float32), y.astype(np.float32)
    clf = curfr(max_depth=3, random_state=0, n_estimators=10)
    clf.fit(X, y)
    expected_pred = clf.predict(X).reshape((-1, 1, 1))

    checkpoint_path = os.path.join(tmpdir, "checkpoint.tl")
    clf.convert_to_treelite_model().to_treelite_checkpoint(checkpoint_path)

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
    clf.convert_to_treelite_model().to_treelite_checkpoint(checkpoint_path)

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
    clf.convert_to_treelite_model().to_treelite_checkpoint(checkpoint_path)

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

    # Does not error
    clf.fit(X, y)


# TODO: Remove in v24.08
@pytest.mark.parametrize(
    "Estimator",
    [
        curfr,
        curfc,
    ],
)
def test_random_forest_max_features_deprecation(Estimator):
    X = np.array([[1.0, 2], [3, 4]])
    y = np.array([1, 0])
    est = Estimator(max_features="auto")

    error_msg = "`max_features='auto'` has been deprecated in 24.06 "

    with pytest.warns(FutureWarning, match=error_msg):
        est.fit(X, y)


def test_rf_predict_returns_int():

    X, y = make_classification()
    clf = cuml.ensemble.RandomForestClassifier().fit(X, y)
    pred = clf.predict(X)
    assert pred.dtype == np.int64

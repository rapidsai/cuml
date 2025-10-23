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

import os
from contextlib import nullcontext
from math import ceil

import cupy as cp
import numpy as np
import pandas as pd
import pytest
import treelite

# Import XGBoost before scikit-learn to work around a libgomp bug
# See https://github.com/dmlc/xgboost/issues/7110
xgb = pytest.importorskip("xgboost")

from sklearn.datasets import make_classification, make_regression  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.model_selection import train_test_split  # noqa: E402

from cuml import ForestInference  # noqa: E402
from cuml.ensemble import (  # noqa: E402
    RandomForestClassifier as cumlRandomForestClassifier,
)
from cuml.fil import get_fil_device_type, set_fil_device_type  # noqa: E402
from cuml.internals.device_type import DeviceType  # noqa: E402
from cuml.internals.global_settings import GlobalSettings  # noqa: E402
from cuml.testing.utils import (  # noqa: E402
    quality_param,
    stress_param,
    unit_param,
)


def simulate_data(
    m,
    n,
    k=2,
    n_informative="auto",
    random_state=None,
    classification=True,
    bias=0.0,
):
    if n_informative == "auto":
        n_informative = n // 5
    if classification:
        features, labels = make_classification(
            n_samples=m,
            n_features=n,
            n_informative=n_informative,
            n_redundant=n - n_informative,
            n_classes=k,
            random_state=random_state,
        )
    else:
        features, labels = make_regression(
            n_samples=m,
            n_features=n,
            n_informative=n_informative,
            n_targets=1,
            bias=bias,
            random_state=random_state,
        )
    return (
        np.c_[features].astype(np.float32),
        np.c_[labels].astype(np.float32).flatten(),
    )


# absolute tolerance for FIL predict_proba
# False is binary classification, True is multiclass
proba_atol = {False: 3e-7, True: 3e-6}


def _build_and_save_xgboost(
    model_path,
    X_train,
    y_train,
    classification=True,
    num_rounds=5,
    n_classes=2,
    xgboost_params=None,
):
    """Trains a small xgboost classifier and saves it to model_path"""
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # instantiate params
    params = {"eval_metric": "error", "max_depth": 25, "device": "cuda"}

    # learning task params
    if classification:
        if n_classes == 2:
            params["objective"] = "binary:logistic"
        else:
            params["num_class"] = n_classes
            params["objective"] = "multi:softprob"
    else:
        params["objective"] = "reg:squarederror"
        params["base_score"] = 0.0

    xgboost_params = {} if xgboost_params is None else xgboost_params
    params.update(xgboost_params)
    bst = xgb.train(params, dtrain, num_rounds)
    bst.save_model(model_path)
    return bst


@pytest.fixture
def reset_fil_device_type():
    """Ensures fil_device_type is reset after a test changing it"""
    orig = GlobalSettings().fil_device_type
    yield
    GlobalSettings().fil_device_type = orig


@pytest.mark.parametrize("device_type", ["gpu", "cpu"])
def test_set_fil_device_type(device_type, reset_fil_device_type):
    set_fil_device_type(device_type)
    assert get_fil_device_type() is DeviceType.from_str(device_type)


@pytest.mark.parametrize("device_type", ["gpu", "cpu"])
def test_set_fil_device_type_context(device_type, reset_fil_device_type):
    orig = get_fil_device_type()
    with set_fil_device_type(device_type):
        assert get_fil_device_type() is DeviceType.from_str(device_type)
    assert get_fil_device_type() is orig


def test_set_fil_device_type_context_nested(reset_fil_device_type):
    orig = get_fil_device_type()
    with set_fil_device_type("gpu"):
        assert get_fil_device_type() is DeviceType.device
        with set_fil_device_type("cpu"):
            assert get_fil_device_type() is DeviceType.host
        assert get_fil_device_type() is DeviceType.device
    assert get_fil_device_type() is orig


@pytest.mark.parametrize("train_device", ("cpu", "gpu"))
@pytest.mark.parametrize("infer_device", ("cpu", "gpu"))
@pytest.mark.parametrize(
    "n_rows", [unit_param(1000), quality_param(10000), stress_param(500000)]
)
@pytest.mark.parametrize(
    "n_columns", [unit_param(30), quality_param(100), stress_param(1000)]
)
@pytest.mark.parametrize(
    "num_rounds",
    [unit_param(1), unit_param(5), quality_param(50), stress_param(90)],
)
@pytest.mark.parametrize("n_classes", [2, 5, 25])
def test_fil_classification(
    train_device,
    infer_device,
    n_rows,
    n_columns,
    num_rounds,
    n_classes,
    tmp_path,
):
    with set_fil_device_type(train_device):
        classification = True  # change this to false to use regression
        random_state = np.random.RandomState(43210)

        X, y = simulate_data(
            n_rows,
            n_columns,
            n_classes,
            random_state=random_state,
            classification=classification,
        )

        X_train, X_validation, y_train, y_validation = train_test_split(
            X, y, train_size=0.8, random_state=0
        )

        model_path = os.path.join(tmp_path, "xgb_class.ubj")

        bst = _build_and_save_xgboost(
            model_path,
            X_train,
            y_train,
            num_rounds=num_rounds,
            classification=classification,
            n_classes=n_classes,
        )

        dvalidation = xgb.DMatrix(X_validation, label=y_validation)
        xgb_proba = bst.predict(dvalidation)

        fm = ForestInference.load(model_path, is_classifier=True)
    with set_fil_device_type(infer_device):
        fil_proba = np.asarray(fm.predict_proba(X_validation))
        fil_proba = np.reshape(fil_proba, xgb_proba.shape)
        fm.optimize(batch_size=len(X_validation))
        fil_proba_opt = np.asarray(fm.predict_proba(X_validation))
        fil_proba_opt = np.reshape(fil_proba_opt, xgb_proba.shape)

        np.testing.assert_almost_equal(fil_proba, xgb_proba, decimal=6)
        np.testing.assert_almost_equal(fil_proba_opt, fil_proba, decimal=6)


@pytest.mark.parametrize("train_device", ("cpu", "gpu"))
@pytest.mark.parametrize("infer_device", ("cpu", "gpu"))
@pytest.mark.parametrize(
    "n_rows", [unit_param(1000), quality_param(10000), stress_param(500000)]
)
@pytest.mark.parametrize(
    "n_columns", [unit_param(20), quality_param(100), stress_param(1000)]
)
@pytest.mark.parametrize(
    "num_rounds", [unit_param(5), quality_param(10), stress_param(90)]
)
@pytest.mark.parametrize(
    "max_depth", [unit_param(3), unit_param(7), stress_param(11)]
)
def test_fil_regression(
    train_device,
    infer_device,
    n_rows,
    n_columns,
    num_rounds,
    tmp_path,
    max_depth,
):
    with set_fil_device_type(train_device):
        classification = False
        random_state = np.random.RandomState(43210)

        X, y = simulate_data(
            n_rows,
            n_columns,
            random_state=random_state,
            classification=classification,
            bias=10.0,
        )
        train_size = 0.80

        X_train, X_validation, y_train, y_validation = train_test_split(
            X, y, train_size=train_size, random_state=0
        )

        model_path = os.path.join(tmp_path, "xgb_reg.ubj")
        bst = _build_and_save_xgboost(
            model_path,
            X_train,
            y_train,
            classification=classification,
            num_rounds=num_rounds,
            xgboost_params={"max_depth": max_depth},
        )

        dvalidation = xgb.DMatrix(X_validation, label=y_validation)
        xgb_preds = bst.predict(dvalidation)

        fm = ForestInference.load(
            path=model_path, is_classifier=False, precision="single"
        )
    with set_fil_device_type(infer_device):
        fil_preds = np.asarray(fm.predict(X_validation))
        fil_preds = np.reshape(fil_preds, np.shape(xgb_preds))
        fm.optimize(data=X_validation)
        fil_preds_opt = np.asarray(fm.predict(X_validation))
        fil_preds_opt = np.reshape(fil_preds_opt, np.shape(xgb_preds))

        np.testing.assert_almost_equal(fil_preds, xgb_preds, decimal=4)
        np.testing.assert_almost_equal(fil_preds_opt, fil_preds, decimal=4)


@pytest.mark.parametrize("train_device", ("cpu", "gpu"))
@pytest.mark.parametrize("infer_device", ("cpu", "gpu"))
@pytest.mark.parametrize("n_rows", [1000])
@pytest.mark.parametrize("n_columns", [30])
@pytest.mark.parametrize(
    "max_depth", [unit_param(2), quality_param(10), stress_param(20)]
)
# When n_classes=25, fit a single estimator only to reduce test time
@pytest.mark.parametrize(
    "n_classes,model_class,n_estimators,precision",
    [
        (2, GradientBoostingClassifier, 1, "native"),
        (2, GradientBoostingClassifier, 10, "native"),
        (2, RandomForestClassifier, 1, "native"),
        (5, RandomForestClassifier, 1, "native"),
        (2, RandomForestClassifier, 10, "native"),
        (5, RandomForestClassifier, 10, "native"),
        (2, ExtraTreesClassifier, 1, "native"),
        (2, ExtraTreesClassifier, 10, "native"),
        (5, GradientBoostingClassifier, 1, "native"),
        (5, GradientBoostingClassifier, 10, "native"),
        (25, GradientBoostingClassifier, 1, "native"),
        (25, RandomForestClassifier, 1, "native"),
        (2, RandomForestClassifier, 10, "float32"),
        (2, RandomForestClassifier, 10, "float64"),
        (5, RandomForestClassifier, 10, "float32"),
        (5, RandomForestClassifier, 10, "float64"),
    ],
)
def test_fil_skl_classification(
    train_device,
    infer_device,
    n_rows,
    n_columns,
    n_estimators,
    max_depth,
    n_classes,
    precision,
    model_class,
):
    with set_fil_device_type(train_device):
        classification = True
        random_state = np.random.RandomState(43210)

        X, y = simulate_data(
            n_rows,
            n_columns,
            n_classes,
            random_state=random_state,
            classification=classification,
        )
        # identify shape and indices
        train_size = 0.80

        X_train, X_validation, y_train, y_validation = train_test_split(
            X, y, train_size=train_size, random_state=0
        )

        init_kwargs = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
        }
        if model_class in [RandomForestClassifier, ExtraTreesClassifier]:
            init_kwargs["max_features"] = 0.3
            init_kwargs["n_jobs"] = -1
        else:
            # model_class == GradientBoostingClassifier
            init_kwargs["init"] = "zero"

        skl_model = model_class(**init_kwargs, random_state=random_state)
        skl_model.fit(X_train, y_train)

        skl_proba = skl_model.predict_proba(X_validation)

        fm = ForestInference.load_from_sklearn(
            skl_model, precision=precision, is_classifier=True
        )
    with set_fil_device_type(infer_device):
        fil_proba = np.asarray(fm.predict_proba(X_validation))
        # Given a binary GradientBoostingClassifier,
        # FIL produces the probability score only for the positive class,
        # whereas scikit-learn produces the probability scores for both
        # the positive and negative class. So we have to transform
        # fil_proba to compare it with skl_proba.
        if n_classes == 2 and model_class == GradientBoostingClassifier:
            fil_proba = np.stack([1 - fil_proba, fil_proba], axis=1)
        fil_proba = np.reshape(fil_proba, skl_proba.shape)

        fm.optimize(data=np.expand_dims(X_validation, 0))
        fil_proba_opt = np.asarray(fm.predict_proba(X_validation))
        if n_classes == 2 and model_class == GradientBoostingClassifier:
            fil_proba_opt = np.stack(
                [1 - fil_proba_opt, fil_proba_opt], axis=1
            )
        fil_proba_opt = np.reshape(fil_proba_opt, skl_proba.shape)
        np.testing.assert_allclose(
            fil_proba, skl_proba, atol=proba_atol[n_classes > 2]
        )
        np.testing.assert_allclose(
            fil_proba_opt, fil_proba, atol=proba_atol[n_classes > 2]
        )


@pytest.mark.parametrize("train_device", ("cpu", "gpu"))
@pytest.mark.parametrize("infer_device", ("cpu", "gpu"))
@pytest.mark.parametrize("n_rows", [1000])
@pytest.mark.parametrize("n_columns", [20])
@pytest.mark.parametrize(
    "max_depth", [unit_param(2), quality_param(10), stress_param(20)]
)
@pytest.mark.parametrize(
    "n_classes,model_class,n_estimators",
    [
        (1, GradientBoostingRegressor, 1),
        (1, GradientBoostingRegressor, 10),
        (1, RandomForestRegressor, 1),
        (1, RandomForestRegressor, 10),
        (5, RandomForestRegressor, 1),
        (5, RandomForestRegressor, 10),
        (1, ExtraTreesRegressor, 1),
        (1, ExtraTreesRegressor, 10),
        (5, GradientBoostingRegressor, 10),
    ],
)
def test_fil_skl_regression(
    train_device,
    infer_device,
    n_rows,
    n_columns,
    n_classes,
    model_class,
    n_estimators,
    max_depth,
):

    with set_fil_device_type(train_device):
        random_state = np.random.RandomState(43210)

        X, y = simulate_data(
            n_rows,
            n_columns,
            n_classes,
            random_state=random_state,
            classification=False,
        )
        # identify shape and indices
        train_size = 0.80

        X_train, X_validation, y_train, y_validation = train_test_split(
            X, y, train_size=train_size, random_state=0
        )

        init_kwargs = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
        }
        if model_class in [RandomForestRegressor, ExtraTreesRegressor]:
            init_kwargs["max_features"] = 0.3
            init_kwargs["n_jobs"] = -1

        skl_model = model_class(**init_kwargs)
        skl_model.fit(X_train, y_train)

        skl_preds = skl_model.predict(X_validation)

        fm = ForestInference.load_from_sklearn(
            skl_model=skl_model,
            is_classifier=False,
            precision="double",
        )
    with set_fil_device_type(infer_device):
        fil_preds = np.asarray(fm.predict(X_validation))
        fil_preds = np.reshape(fil_preds, np.shape(skl_preds))
        fm.optimize(batch_size=len(X_validation))
        fil_preds_opt = np.asarray(fm.predict(X_validation))
        fil_preds_opt = np.reshape(fil_preds_opt, np.shape(skl_preds))

        np.testing.assert_almost_equal(fil_preds, skl_preds)
        np.testing.assert_almost_equal(fil_preds_opt, fil_preds)


@pytest.fixture(scope="session", params=["ubjson", "json"])
def small_classifier_and_preds(tmpdir_factory, request):
    X, y = simulate_data(500, 10, random_state=43210, classification=True)

    ext = "json" if request.param == "json" else "ubj"
    model_type = "xgboost_json" if request.param == "json" else "xgboost_ubj"
    model_path = str(
        tmpdir_factory.mktemp("models").join(f"small_class.{ext}")
    )
    bst = _build_and_save_xgboost(model_path, X, y)
    # just do within-sample since it's not an accuracy test
    dtrain = xgb.DMatrix(X, label=y)
    xgb_preds = bst.predict(dtrain)

    return model_path, model_type, X, xgb_preds


@pytest.mark.parametrize("train_device", ("cpu", "gpu"))
@pytest.mark.parametrize("infer_device", ("cpu", "gpu"))
@pytest.mark.parametrize("precision", ["native", "float32", "float64"])
def test_precision_xgboost(
    train_device, infer_device, precision, small_classifier_and_preds
):
    with set_fil_device_type(train_device):
        model_path, model_type, X, xgb_preds = small_classifier_and_preds
        fm = ForestInference.load(
            model_path,
            model_type=model_type,
            is_classifier=True,
            precision=precision,
        )

    with set_fil_device_type(infer_device):
        fil_preds = np.asarray(fm.predict_proba(X))
        fil_preds = np.reshape(fil_preds, xgb_preds.shape)

        np.testing.assert_almost_equal(fil_preds, xgb_preds)


@pytest.mark.parametrize("train_device", ("cpu", "gpu"))
@pytest.mark.parametrize("infer_device", ("cpu", "gpu"))
@pytest.mark.parametrize("layout", ["depth_first", "breadth_first", "layered"])
@pytest.mark.parametrize("chunk_size", [2, 4, 8, 16, 32])
def test_performance_hyperparameters(
    train_device, infer_device, layout, chunk_size, small_classifier_and_preds
):
    with set_fil_device_type(train_device):
        model_path, model_type, X, xgb_preds = small_classifier_and_preds
        fm = ForestInference.load(
            model_path,
            layout=layout,
            is_classifier=True,
            model_type=model_type,
        )

    with set_fil_device_type(infer_device):
        fil_proba = np.asarray(fm.predict_proba(X, chunk_size=chunk_size))
        fil_proba = np.reshape(fil_proba, xgb_preds.shape)

        np.testing.assert_almost_equal(fil_proba, xgb_preds)


@pytest.mark.parametrize("chunk_size", [2, 4, 8, 16, 32, 64, 128, 256])
def test_chunk_size(chunk_size, small_classifier_and_preds):
    model_path, model_type, X, xgb_preds = small_classifier_and_preds
    fm = ForestInference.load(
        model_path,
        model_type=model_type,
        is_classifier=True,
    )

    fil_preds = np.asarray(fm.predict(X, chunk_size=chunk_size))
    fil_proba = np.asarray(
        fm.predict_proba(X, chunk_size=chunk_size)
    ).squeeze()

    np.testing.assert_almost_equal(fil_proba, xgb_preds)

    xgb_preds_int = np.around(xgb_preds)
    fil_preds = np.reshape(fil_preds, np.shape(xgb_preds_int))
    np.testing.assert_array_equal(fil_preds, xgb_preds_int)


@pytest.mark.parametrize("train_device", ("cpu", "gpu"))
@pytest.mark.parametrize("infer_device", ("cpu", "gpu"))
def test_output_args(train_device, infer_device, small_classifier_and_preds):
    with set_fil_device_type(train_device):
        model_path, model_type, X, xgb_preds = small_classifier_and_preds
        fm = ForestInference.load(
            model_path, is_classifier=True, model_type=model_type
        )
    with set_fil_device_type(infer_device):
        X = np.asarray(X)
        fil_preds = np.asarray(fm.predict_proba(X))
        fil_preds = np.reshape(fil_preds, np.shape(xgb_preds))

    np.testing.assert_almost_equal(fil_preds, xgb_preds)


def to_categorical(features, n_categorical, invalid_frac, random_state):
    """returns data in two formats: pandas (for LightGBM) and numpy (for FIL)
    LightGBM needs a DataFrame to recognize and fit on categorical columns.
    Second fp32 output is to test invalid categories for prediction only.
    """
    features = features.copy()  # avoid clobbering source matrix
    rng = np.random.default_rng(hash(random_state))  # allow RandomState object
    # the main bottleneck (>80%) of to_categorical() is the pandas operations
    n_features = features.shape[1]
    # all categorical columns
    cat_cols = features[:, :n_categorical]
    # axis=1 means 0th dimension remains. Row-major FIL means 0th dimension is
    # the number of columns. We reduce within columns, across rows.
    cat_cols = cat_cols - cat_cols.min(axis=0, keepdims=True)  # range [0, ?]
    cat_cols /= cat_cols.max(axis=0, keepdims=True)  # range [0, 1]
    rough_n_categories = 100
    # round into rough_n_categories bins
    cat_cols = (cat_cols * rough_n_categories).astype(int)

    # mix categorical and numerical columns
    new_col_idx = rng.choice(
        n_features, n_features, replace=False, shuffle=True
    )
    df_cols = {}
    for icol in range(n_categorical):
        col = cat_cols[:, icol]
        df_cols[new_col_idx[icol]] = pd.Series(
            pd.Categorical(col, categories=np.unique(col))
        )
    # all numerical columns
    for icol in range(n_categorical, n_features):
        df_cols[new_col_idx[icol]] = pd.Series(features[:, icol])
    fit_df = pd.DataFrame(df_cols)

    # randomly inject invalid categories only into predict_matrix
    invalid_idx = rng.choice(
        a=cat_cols.size,
        size=ceil(cat_cols.size * invalid_frac),
        replace=False,
        shuffle=False,
    )
    cat_cols.flat[invalid_idx] += rough_n_categories
    # mix categorical and numerical columns
    predict_matrix = np.concatenate(
        [cat_cols, features[:, n_categorical:]], axis=1
    )
    predict_matrix[:, new_col_idx] = predict_matrix

    return fit_df, predict_matrix


@pytest.mark.parametrize("train_device", ("cpu", "gpu"))
@pytest.mark.parametrize("infer_device", ("cpu", "gpu"))
@pytest.mark.parametrize("num_classes", [2, 5])
@pytest.mark.parametrize("n_categorical", [0, 5])
def test_lightgbm(
    train_device, infer_device, tmp_path, num_classes, n_categorical
):
    lgb = pytest.importorskip("lightgbm")

    if n_categorical > 0:
        n_features = 10
        n_rows = 1000
        n_informative = n_features
    else:
        n_features = 10 if num_classes == 2 else 50
        n_rows = 500
        n_informative = "auto"

    X, y = simulate_data(
        n_rows,
        n_features,
        num_classes,
        n_informative=n_informative,
        random_state=43210,
        classification=True,
    )
    if n_categorical > 0:
        X_fit, X_predict = to_categorical(
            X,
            n_categorical=n_categorical,
            invalid_frac=0.1,
            random_state=43210,
        )
    else:
        X_fit, X_predict = X, X

    model_path = str(os.path.join(tmp_path, "lgb.model"))

    lgm = lgb.LGBMClassifier(
        objective="multiclass" if num_classes > 2 else "binary",
        boosting_type="gbdt",
        n_estimators=5,
    )
    lgm.fit(X_fit, y)
    lgm.booster_.save_model(model_path)
    with set_fil_device_type(train_device):
        fm = ForestInference.load(
            model_path, is_classifier=True, model_type="lightgbm"
        )
    gbm_proba = lgm.predict_proba(X_predict)
    with set_fil_device_type(infer_device):
        fil_proba = fm.predict_proba(X_predict)
    # Given a binary classifier, FIL produces the probability score
    # only for the positive class,
    # whereas LGBMClassifier produces the probability scores for both
    # the positive and negative class. So we have to transform
    # fil_proba to compare it with gbm_proba.
    if num_classes == 2:
        fil_proba = np.concatenate([1 - fil_proba, fil_proba], axis=1)
    np.testing.assert_almost_equal(gbm_proba, fil_proba)


@pytest.mark.parametrize("train_device", ("cpu", "gpu"))
@pytest.mark.parametrize("infer_device", ("cpu", "gpu"))
@pytest.mark.parametrize("n_classes", [2, 5, 25])
@pytest.mark.parametrize("num_boost_round", [10, 100])
def test_predict_per_tree(
    train_device, infer_device, n_classes, num_boost_round, tmp_path
):
    n_rows = 1000
    n_columns = 30

    with set_fil_device_type(train_device):
        X, y = simulate_data(
            n_rows,
            n_columns,
            n_classes,
            random_state=0,
            classification=True,
        )

        model_path = os.path.join(tmp_path, "xgb_class.ubj")

        xgboost_params = {"base_score": (0.5 if n_classes == 2 else 0.0)}
        bst = _build_and_save_xgboost(
            model_path,
            X,
            y,
            num_rounds=num_boost_round,
            classification=True,
            n_classes=n_classes,
            xgboost_params=xgboost_params,
        )
        fm = ForestInference.load(model_path, is_classifier=True)
        tl_model = treelite.frontend.from_xgboost(bst)
        pred_per_tree_tl = treelite.gtil.predict_per_tree(tl_model, X)

    with set_fil_device_type(infer_device):
        pred_per_tree = fm.predict_per_tree(X)
        margin_pred = bst.predict(xgb.DMatrix(X), output_margin=True)
        if n_classes == 2:
            expected_shape = (n_rows, num_boost_round)
            sum_by_class = np.sum(pred_per_tree, axis=1)
        else:
            expected_shape = (n_rows, num_boost_round * n_classes)
            sum_by_class = np.column_stack(
                tuple(
                    np.sum(pred_per_tree[:, class_id::n_classes], axis=1)
                    for class_id in range(n_classes)
                )
            )
        fm.optimize(batch_size=len(X), predict_method="predict_per_tree")
        pred_per_tree_opt = fm.predict_per_tree(X)
        assert pred_per_tree.shape == expected_shape
        np.testing.assert_almost_equal(sum_by_class, margin_pred, decimal=3)
        np.testing.assert_almost_equal(
            pred_per_tree.reshape((n_rows, -1, 1)), pred_per_tree_tl, decimal=3
        )
        np.testing.assert_almost_equal(
            pred_per_tree_opt, pred_per_tree, decimal=3
        )


@pytest.mark.parametrize("train_device", ("cpu", "gpu"))
@pytest.mark.parametrize("infer_device", ("cpu", "gpu"))
@pytest.mark.parametrize("n_classes", [5, 25])
def test_predict_per_tree_with_vector_leaf(
    train_device, infer_device, n_classes, tmp_path
):
    n_rows = 1000
    n_columns = 30
    n_estimators = 10

    with set_fil_device_type(train_device):
        X, y = simulate_data(
            n_rows,
            n_columns,
            n_classes,
            random_state=0,
            classification=True,
        )

        skl_model = RandomForestClassifier(
            max_depth=3, random_state=0, n_estimators=n_estimators
        )
        skl_model.fit(X, y)
        tl_model = treelite.sklearn.import_model(skl_model)
        pred_per_tree_tl = treelite.gtil.predict_per_tree(tl_model, X)
        fm = ForestInference.load_from_sklearn(
            skl_model, precision="native", is_classifier=True
        )

    with set_fil_device_type(infer_device):
        pred_per_tree = fm.predict_per_tree(X)
        fm.optimize(batch_size=len(X), predict_method="predict_per_tree")
        pred_per_tree_opt = fm.predict_per_tree(X)
        margin_pred = skl_model.predict_proba(X)
        assert pred_per_tree.shape == (n_rows, n_estimators, n_classes)
        avg_by_class = np.sum(pred_per_tree, axis=1) / n_estimators
        np.testing.assert_almost_equal(avg_by_class, margin_pred, decimal=3)
        np.testing.assert_almost_equal(
            pred_per_tree, pred_per_tree_tl, decimal=3
        )
        np.testing.assert_almost_equal(
            pred_per_tree_opt, pred_per_tree, decimal=3
        )


@pytest.mark.parametrize("train_device", ("cpu", "gpu"))
@pytest.mark.parametrize("infer_device", ("cpu", "gpu"))
@pytest.mark.parametrize("n_classes", [2, 5, 25])
def test_apply(train_device, infer_device, n_classes, tmp_path):
    n_rows = 1000
    n_columns = 30
    num_boost_round = 10

    with set_fil_device_type(train_device):
        X, y = simulate_data(
            n_rows,
            n_columns,
            n_classes,
            random_state=0,
            classification=True,
        )

        model_path = os.path.join(tmp_path, "xgb_class.ubj")

        xgboost_params = {"base_score": (0.5 if n_classes == 2 else 0.0)}
        bst = _build_and_save_xgboost(
            model_path,
            X,
            y,
            num_rounds=num_boost_round,
            classification=True,
            n_classes=n_classes,
            xgboost_params=xgboost_params,
        )

        fm = ForestInference.load(
            model_path, is_classifier=True, model_type="xgboost_ubj"
        )

    with set_fil_device_type(infer_device):
        pred_leaf = fm.apply(X).astype(np.int32)
        expected_pred_leaf = bst.predict(xgb.DMatrix(X), pred_leaf=True)
        if n_classes == 2:
            expected_shape = (n_rows, num_boost_round)
        else:
            expected_shape = (n_rows, num_boost_round * n_classes)
        assert pred_leaf.shape == expected_shape
        np.testing.assert_equal(pred_leaf, expected_pred_leaf)


@pytest.mark.parametrize("category_list", [[], [0, 2]])
def test_missing_categorical(category_list):
    builder = treelite.model_builder.ModelBuilder(
        threshold_type="float32",
        leaf_output_type="float32",
        metadata=treelite.model_builder.Metadata(
            num_feature=1,
            task_type="kBinaryClf",
            average_tree_output=False,
            num_target=1,
            num_class=[1],
            leaf_vector_shape=(1, 1),
        ),
        tree_annotation=treelite.model_builder.TreeAnnotation(
            num_tree=1, target_id=[0], class_id=[0]
        ),
        postprocessor=treelite.model_builder.PostProcessorFunc(
            name="identity"
        ),
        base_scores=[0.0],
    )
    builder.start_tree()
    builder.start_node(0)
    builder.categorical_test(
        feature_id=0,
        category_list=category_list,
        default_left=False,
        category_list_right_child=False,
        left_child_key=1,
        right_child_key=2,
    )
    builder.end_node()
    builder.start_node(1)
    builder.leaf(1.0)
    builder.end_node()
    builder.start_node(2)
    builder.leaf(2.0)
    builder.end_node()
    builder.end_tree()

    model = builder.commit()

    input = np.array([[np.nan]])
    gtil_preds = treelite.gtil.predict(model, input)
    fm = ForestInference.load_from_treelite_model(model)
    fil_preds = np.asarray(fm.predict(input))
    np.testing.assert_equal(fil_preds.flatten(), gtil_preds.flatten())


@pytest.mark.parametrize("device_id", [None, 0, 1, 2])
@pytest.mark.parametrize("model_kind", ["sklearn", "xgboost", "cuml"])
def test_device_selection(device_id, model_kind, tmp_path):
    current_device = cp.cuda.runtime.getDevice()

    if device_id is not None and device_id >= cp.cuda.runtime.getDeviceCount():
        pytest.skip(
            reason="device_id larger than the number of available GPU devices"
        )

    n_rows = 1000
    n_columns = 30
    n_classes = 3
    n_estimators = 10

    X, y = simulate_data(
        n_rows,
        n_columns,
        n_classes,
        random_state=0,
        classification=True,
    )

    # 1. Model can be loaded with device_id set
    if model_kind == "sklearn":
        skl_model = RandomForestClassifier(
            max_depth=3, random_state=0, n_estimators=n_estimators
        )
        skl_model.fit(X, y)
        fm = ForestInference.load_from_sklearn(
            skl_model,
            precision="native",
            is_classifier=True,
            device_id=device_id,
        )
    elif model_kind == "xgboost":
        xgb_model = xgb.XGBClassifier(
            max_depth=3, random_state=0, n_estimators=n_estimators
        )
        xgb_model.fit(X, y)
        model_path = os.path.join(tmp_path, "xgb_class.ubj")
        xgb_model.save_model(model_path)
        fm = ForestInference.load(
            model_path,
            model_type="xgboost_ubj",
            precision="native",
            is_classifier=True,
            device_id=device_id,
        )
    elif model_kind == "cuml":
        device_context = (
            cp.cuda.Device(device_id) if device_id else nullcontext()
        )

        with device_context:
            # TODO(hcho3): Remove n_streams=1 argument once the bug
            # https://github.com/rapidsai/cuml/issues/5983 is resolved
            cuml_model = cumlRandomForestClassifier(
                max_depth=3,
                random_state=0,
                n_estimators=n_estimators,
                n_streams=1,
            )
            cuml_model.fit(cp.array(X), cp.array(y))
            fm = cuml_model.as_fil()
    else:
        raise NotImplementedError()

    # 2. The section above didn't corrupt current device context
    assert cp.cuda.runtime.getDevice() == current_device

    # 3. Device selection is correctly saved to device_id property
    assert fm.device_id == (device_id if device_id else 0)

    # 4. Inference can run on an input with the selected device
    device_context = cp.cuda.Device(device_id) if device_id else nullcontext()
    with device_context:
        _ = fm.predict_proba(cp.array(X))

    # 5. The section above didn't corrupt current device context
    assert cp.cuda.runtime.getDevice() == current_device

    # 6. Attempting to run inference with an input from a different device
    #    is an error
    if device_id is not None and device_id != 0:
        with cp.cuda.Device(0), pytest.raises(
            RuntimeError, match=r".*I/O data on different device than model.*"
        ):
            _ = fm.predict_proba(cp.array(X))

    # 7. The section above didn't corrupt current device context
    assert cp.cuda.runtime.getDevice() == current_device


def test_wide_data():
    n_rows = 50
    n_features = 100000
    X = np.random.normal(size=(n_rows, n_features)).astype(np.float32)
    y = np.asarray([0, 1] * (n_rows // 2), dtype=np.int32)

    clf = RandomForestClassifier(max_features="sqrt", n_estimators=10)
    clf.fit(X, y)

    # Inference should run without crashing
    fm = ForestInference.load_from_sklearn(clf, is_classifier=True)
    _ = fm.predict(X)

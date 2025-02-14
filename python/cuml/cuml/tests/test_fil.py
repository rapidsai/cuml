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

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
from sklearn.datasets import make_classification, make_regression
from cuml.internals.import_utils import has_xgboost
from cuml.testing.utils import (
    array_equal,
    unit_param,
    quality_param,
    stress_param,
)
from cuml import ForestInference
from math import ceil
import os
import pytest
from cuml.internals.safe_imports import cpu_only_import

np = cpu_only_import("numpy")
pd = cpu_only_import("pandas")

# from cuml.internals.import_utils import has_lightgbm


if has_xgboost():
    import xgboost as xgb

# pytestmark = pytest.mark.skip


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
    xgboost_params={},
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

    params.update(xgboost_params)
    bst = xgb.train(params, dtrain, num_rounds)
    bst.save_model(model_path)
    return bst


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
@pytest.mark.skipif(has_xgboost() is False, reason="need to install xgboost")
def test_fil_classification(
    n_rows, n_columns, num_rounds, n_classes, tmp_path
):
    # settings
    classification = True  # change this to false to use regression
    random_state = np.random.RandomState(43210)

    X, y = simulate_data(
        n_rows,
        n_columns,
        n_classes,
        random_state=random_state,
        classification=classification,
    )
    # identify shape and indices
    n_rows, n_columns = X.shape
    train_size = 0.80

    X_train, X_validation, y_train, y_validation = train_test_split(
        X, y, train_size=train_size, random_state=0
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

    if n_classes == 2:
        xgb_preds = bst.predict(dvalidation)
        xgb_preds_int = np.around(xgb_preds)
        xgb_proba = np.stack([1 - xgb_preds, xgb_preds], axis=1)
    else:
        xgb_proba = bst.predict(dvalidation)
        xgb_preds_int = xgb_proba.argmax(axis=1)
    xgb_acc = accuracy_score(y_validation, xgb_preds_int)

    fm = ForestInference.load(
        model_path, algo="auto", output_class=True, threshold=0.50
    )
    fil_preds = np.asarray(fm.predict(X_validation))
    fil_proba = np.asarray(fm.predict_proba(X_validation))
    fil_acc = accuracy_score(y_validation, fil_preds)

    assert fil_acc == pytest.approx(xgb_acc, abs=0.01)
    assert array_equal(fil_preds, xgb_preds_int)
    np.testing.assert_allclose(
        fil_proba, xgb_proba, atol=proba_atol[n_classes > 2]
    )


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
@pytest.mark.skipif(has_xgboost() is False, reason="need to install xgboost")
def test_fil_regression(n_rows, n_columns, num_rounds, tmp_path, max_depth):
    # settings
    classification = False  # change this to false to use regression
    n_rows = n_rows  # we'll use 1 millions rows
    n_columns = n_columns
    random_state = np.random.RandomState(43210)

    X, y = simulate_data(
        n_rows,
        n_columns,
        random_state=random_state,
        classification=classification,
        bias=10.0,
    )
    # identify shape and indices
    n_rows, n_columns = X.shape
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

    xgb_mse = mean_squared_error(y_validation, xgb_preds)
    fm = ForestInference.load(model_path, algo="auto", output_class=False)
    fil_preds = np.asarray(fm.predict(X_validation))
    fil_preds = np.reshape(fil_preds, np.shape(xgb_preds))
    fil_mse = mean_squared_error(y_validation, fil_preds)

    assert fil_mse == pytest.approx(xgb_mse, abs=0.01)
    assert np.allclose(fil_preds, xgb_preds, 1e-3)


@pytest.mark.parametrize("n_rows", [1000])
@pytest.mark.parametrize("n_columns", [30])
# Skip depth 20 for dense tests
@pytest.mark.parametrize(
    "max_depth,storage_type",
    [(2, False), (2, True), (10, False), (10, True), (20, True)],
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
    n_rows,
    n_columns,
    n_estimators,
    max_depth,
    n_classes,
    storage_type,
    precision,
    model_class,
):
    # settings
    classification = True  # change this to false to use regression
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

    skl_preds = skl_model.predict(X_validation)
    skl_preds_int = np.around(skl_preds)
    skl_proba = skl_model.predict_proba(X_validation)

    skl_acc = accuracy_score(y_validation, skl_preds_int)

    algo = "NAIVE" if storage_type else "BATCH_TREE_REORG"

    fm = ForestInference.load_from_sklearn(
        skl_model,
        algo=algo,
        output_class=True,
        threshold=0.50,
        storage_type=storage_type,
        precision=precision,
    )
    fil_preds = np.asarray(fm.predict(X_validation))
    fil_preds = np.reshape(fil_preds, np.shape(skl_preds_int))
    fil_acc = accuracy_score(y_validation, fil_preds)
    # fil_acc is within p99 error bars of skl_acc (diff == 0.017 +- 0.012)
    # however, some tests have a delta as big as 0.04.
    # sklearn uses float64 thresholds, while FIL uses float32
    # TODO(levsnv): once FIL supports float64 accuracy, revisit thresholds
    threshold = 1e-5 if n_classes == 2 else 0.1
    assert fil_acc == pytest.approx(skl_acc, abs=threshold)

    if n_classes == 2:
        assert array_equal(fil_preds, skl_preds_int)
    fil_proba = np.asarray(fm.predict_proba(X_validation))
    fil_proba = np.reshape(fil_proba, np.shape(skl_proba))
    np.testing.assert_allclose(
        fil_proba, skl_proba, atol=proba_atol[n_classes > 2]
    )


@pytest.mark.parametrize("n_rows", [1000])
@pytest.mark.parametrize("n_columns", [20])
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
@pytest.mark.parametrize("max_depth", [2, 10, 20])
@pytest.mark.parametrize("storage_type", [False, True])
@pytest.mark.skip("https://github.com/rapidsai/cuml/issues/5138")
def test_fil_skl_regression(
    n_rows,
    n_columns,
    n_classes,
    model_class,
    n_estimators,
    max_depth,
    storage_type,
):

    # skip depth 20 for dense tests
    if max_depth == 20 and not storage_type:
        return

    # settings
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
    else:
        # model_class == GradientBoostingRegressor
        init_kwargs["init"] = "zero"

    skl_model = model_class(**init_kwargs)
    skl_model.fit(X_train, y_train)

    skl_preds = skl_model.predict(X_validation)

    skl_mse = mean_squared_error(y_validation, skl_preds)

    algo = "NAIVE" if storage_type else "BATCH_TREE_REORG"

    fm = ForestInference.load_from_sklearn(
        skl_model, algo=algo, output_class=False, storage_type=storage_type
    )
    fil_preds = np.asarray(fm.predict(X_validation))
    fil_preds = np.reshape(fil_preds, np.shape(skl_preds))

    fil_mse = mean_squared_error(y_validation, fil_preds)

    # NOTE(wphicks): Tolerance has been temporarily increased from 1.e-6/1e-4
    # to 1e-4/1e-2. This is too high of a tolerance for this test, but we will
    # use it to unblock CI while investigating the underlying issue.
    # https://github.com/rapidsai/cuml/issues/5138
    assert fil_mse <= skl_mse * (1.0 + 1e-4) + 1e-2
    # NOTE(wphicks): Tolerance has been temporarily increased from 1.2e-3 to
    # 1.2e-2. This test began failing CI due to the previous tolerance more
    # regularly, and while the root cause is under investigation
    # (https://github.com/rapidsai/cuml/issues/5138), the tolerance has simply
    # been reduced. Combined with the above assertion, this is still a very
    # reasonable threshold.
    assert np.allclose(fil_preds, skl_preds, 1.2e-2)


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

    return (model_path, model_type, X, xgb_preds)


@pytest.mark.skipif(has_xgboost() is False, reason="need to install xgboost")
@pytest.mark.parametrize(
    "algo",
    [
        "AUTO",
        "NAIVE",
        "TREE_REORG",
        "BATCH_TREE_REORG",
        "auto",
        "naive",
        "tree_reorg",
        "batch_tree_reorg",
    ],
)
def test_output_algos(algo, small_classifier_and_preds):
    model_path, model_type, X, xgb_preds = small_classifier_and_preds
    fm = ForestInference.load(
        model_path,
        model_type=model_type,
        algo=algo,
        output_class=True,
        threshold=0.50,
    )

    xgb_preds_int = np.around(xgb_preds)
    fil_preds = np.asarray(fm.predict(X))
    fil_preds = np.reshape(fil_preds, np.shape(xgb_preds_int))

    assert np.allclose(fil_preds, xgb_preds_int, 1e-3)


@pytest.mark.skipif(has_xgboost() is False, reason="need to install xgboost")
@pytest.mark.parametrize("precision", ["native", "float32", "float64"])
def test_precision_xgboost(precision, small_classifier_and_preds):
    model_path, model_type, X, xgb_preds = small_classifier_and_preds
    fm = ForestInference.load(
        model_path,
        model_type=model_type,
        output_class=True,
        threshold=0.50,
        precision=precision,
    )

    xgb_preds_int = np.around(xgb_preds)
    fil_preds = np.asarray(fm.predict(X))
    fil_preds = np.reshape(fil_preds, np.shape(xgb_preds_int))

    assert np.allclose(fil_preds, xgb_preds_int, 1e-3)


@pytest.mark.skipif(has_xgboost() is False, reason="need to install xgboost")
@pytest.mark.parametrize(
    "storage_type", [False, True, "auto", "dense", "sparse", "sparse8"]
)
def test_output_storage_type(storage_type, small_classifier_and_preds):
    model_path, model_type, X, xgb_preds = small_classifier_and_preds
    fm = ForestInference.load(
        model_path,
        model_type=model_type,
        output_class=True,
        storage_type=storage_type,
        threshold=0.50,
    )

    xgb_preds_int = np.around(xgb_preds)
    fil_preds = np.asarray(fm.predict(X))
    fil_preds = np.reshape(fil_preds, np.shape(xgb_preds_int))

    assert np.allclose(fil_preds, xgb_preds_int, 1e-3)


@pytest.mark.skipif(has_xgboost() is False, reason="need to install xgboost")
@pytest.mark.parametrize("storage_type", ["dense", "sparse"])
@pytest.mark.parametrize("blocks_per_sm", [1, 2, 3, 4])
def test_output_blocks_per_sm(
    storage_type, blocks_per_sm, small_classifier_and_preds
):
    model_path, model_type, X, xgb_preds = small_classifier_and_preds
    fm = ForestInference.load(
        model_path,
        model_type=model_type,
        output_class=True,
        storage_type=storage_type,
        threshold=0.50,
        blocks_per_sm=blocks_per_sm,
    )

    xgb_preds_int = np.around(xgb_preds)
    fil_preds = np.asarray(fm.predict(X))
    fil_preds = np.reshape(fil_preds, np.shape(xgb_preds_int))

    assert np.allclose(fil_preds, xgb_preds_int, 1e-3)


@pytest.mark.skipif(has_xgboost() is False, reason="need to install xgboost")
@pytest.mark.parametrize("threads_per_tree", [2, 4, 8, 16, 32, 64, 128, 256])
def test_threads_per_tree(threads_per_tree, small_classifier_and_preds):
    model_path, model_type, X, xgb_preds = small_classifier_and_preds
    fm = ForestInference.load(
        model_path,
        model_type=model_type,
        output_class=True,
        storage_type="auto",
        threshold=0.50,
        threads_per_tree=threads_per_tree,
        n_items=1,
    )

    fil_preds = np.asarray(fm.predict(X))
    fil_proba = np.asarray(fm.predict_proba(X))

    xgb_proba = np.stack([1 - xgb_preds, xgb_preds], axis=1)
    np.testing.assert_allclose(fil_proba, xgb_proba, atol=proba_atol[False])

    xgb_preds_int = np.around(xgb_preds)
    fil_preds = np.reshape(fil_preds, np.shape(xgb_preds_int))
    assert np.allclose(fil_preds, xgb_preds_int, 1e-3)


@pytest.mark.skipif(has_xgboost() is False, reason="need to install xgboost")
def test_print_forest_shape(small_classifier_and_preds):
    model_path, model_type, X, xgb_preds = small_classifier_and_preds
    m = ForestInference.load(
        model_path,
        model_type=model_type,
        output_class=True,
        compute_shape_str=True,
    )
    for substr in [
        "model size",
        " MB",
        "Depth histogram:",
        "Leaf depth",
        "Depth histogram fingerprint",
        "Avg nodes per tree",
    ]:
        assert substr in m.shape_str


@pytest.mark.parametrize("output_class", [True, False])
@pytest.mark.skipif(has_xgboost() is False, reason="need to install xgboost")
def test_thresholding(output_class, small_classifier_and_preds):
    model_path, model_type, X, xgb_preds = small_classifier_and_preds
    fm = ForestInference.load(
        model_path,
        model_type=model_type,
        algo="TREE_REORG",
        output_class=output_class,
        threshold=0.50,
    )
    fil_preds = np.asarray(fm.predict(X))
    if output_class:
        assert ((fil_preds != 0.0) & (fil_preds != 1.0)).sum() == 0
    else:
        assert ((fil_preds != 0.0) & (fil_preds != 1.0)).sum() > 0


@pytest.mark.skipif(has_xgboost() is False, reason="need to install xgboost")
def test_output_args(small_classifier_and_preds):
    model_path, model_type, X, xgb_preds = small_classifier_and_preds
    fm = ForestInference.load(
        model_path,
        model_type=model_type,
        algo="TREE_REORG",
        output_class=False,
        threshold=0.50,
    )
    X = np.asarray(X)
    fil_preds = fm.predict(X)
    fil_preds = np.reshape(fil_preds, np.shape(xgb_preds))

    assert array_equal(fil_preds, xgb_preds, 1e-3)


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


@pytest.mark.parametrize("num_classes", [2, 5])
@pytest.mark.parametrize("n_categorical", [0, 5])
@pytest.mark.skip(reason="Causing CI to hang.")
# @pytest.mark.skipif(has_lightgbm() is False,
#                     reason="need to install lightgbm")
def test_lightgbm(tmp_path, num_classes, n_categorical):
    import lightgbm as lgb

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

    train_data = lgb.Dataset(X_fit, label=y)
    num_round = 5
    model_path = str(os.path.join(tmp_path, "lgb.model"))

    if num_classes == 2:
        param = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_class": 1,
        }
        bst = lgb.train(param, train_data, num_round)
        bst.save_model(model_path)
        fm = ForestInference.load(
            model_path,
            algo="TREE_REORG",
            output_class=True,
            model_type="lightgbm",
        )
        # binary classification
        gbm_proba = bst.predict(X_predict)
        fil_proba = fm.predict_proba(X_predict)[:, 1]
        gbm_preds = (gbm_proba > 0.5).astype(float)
        fil_preds = fm.predict(X_predict)
        assert array_equal(gbm_preds, fil_preds)
        np.testing.assert_allclose(
            gbm_proba, fil_proba, atol=proba_atol[num_classes > 2]
        )
    else:
        # multi-class classification
        lgm = lgb.LGBMClassifier(
            objective="multiclass",
            boosting_type="gbdt",
            n_estimators=num_round,
        )
        lgm.fit(X_fit, y)
        lgm.booster_.save_model(model_path)
        lgm_preds = lgm.predict(X_predict).astype(int)
        fm = ForestInference.load(
            model_path,
            algo="TREE_REORG",
            output_class=True,
            model_type="lightgbm",
        )
        assert array_equal(
            lgm.booster_.predict(X_predict).argmax(axis=1), lgm_preds
        )
        assert array_equal(lgm_preds, fm.predict(X_predict))
        # lightgbm uses float64 thresholds, while FIL uses float32
        np.testing.assert_allclose(
            lgm.predict_proba(X_predict),
            fm.predict_proba(X_predict),
            atol=proba_atol[num_classes > 2],
        )

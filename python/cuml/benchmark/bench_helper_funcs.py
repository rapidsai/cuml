#
# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
import cuml
from cuml.common import input_utils
import numpy as np
import pandas as pd
import pickle as pickle
import sklearn.ensemble as skl_ensemble
import cudf
from numba import cuda
from cuml.benchmark import datagen
from cuml.manifold import UMAP


def call(m, func_name, X, y=None):
    def unwrap_and_get_args(func):
        if hasattr(func, '__wrapped__'):
            return unwrap_and_get_args(func.__wrapped__)
        else:
            return func.__code__.co_varnames

    if not hasattr(m, func_name):
        raise ValueError('Model does not have function ' + func_name)
    func = getattr(m, func_name)
    argnames = unwrap_and_get_args(func)
    if y is not None and 'y' in argnames:
        func(X, y=y)
    else:
        func(X)


def pass_func(m, x, y=None):
    pass


def fit(m, x, y=None):
    call(m, 'fit', x, y)


def predict(m, x, y=None):
    call(m, 'predict', x)


def transform(m, x, y=None):
    call(m, 'transform', x)


def kneighbors(m, x, y=None):
    call(m, 'kneighbors', x)


def fit_predict(m, x, y=None):
    if hasattr(m, 'predict'):
        fit(m, x, y)
        predict(m, x)
    else:
        call(m, 'fit_predict', x, y)


def fit_transform(m, x, y=None):
    if hasattr(m, 'transform'):
        fit(m, x, y)
        transform(m, x)
    else:
        call(m, 'fit_transform', x, y)


def fit_kneighbors(m, x, y=None):
    if hasattr(m, 'kneighbors'):
        fit(m, x, y)
        kneighbors(m, x)
    else:
        call(m, 'fit_kneighbors', x, y)


def _training_data_to_numpy(X, y):
    """Convert input training data into numpy format"""
    if isinstance(X, np.ndarray):
        X_np = X
        y_np = y
    elif isinstance(X, cudf.DataFrame):
        X_np = X.to_numpy()
        y_np = y.to_numpy()
    elif cuda.devicearray.is_cuda_ndarray(X):
        X_np = X.copy_to_host()
        y_np = y.copy_to_host()
    elif isinstance(X, (pd.DataFrame, pd.Series)):
        X_np = datagen._convert_to_numpy(X)
        y_np = datagen._convert_to_numpy(y)
    else:
        raise TypeError("Received unsupported input type")
    return X_np, y_np


def _build_fil_classifier(m, data, args, tmpdir):
    """Setup function for FIL classification benchmarking"""
    from cuml.common.import_utils import has_xgboost
    if has_xgboost():
        import xgboost as xgb
    else:
        raise ImportError("No XGBoost package found")

    train_data, train_label = _training_data_to_numpy(data[0], data[1])

    dtrain = xgb.DMatrix(train_data, label=train_label)

    params = {
        "silent": 1, "eval_metric": "error",
        "objective": "binary:logistic", "tree_method": "gpu_hist",
    }
    params.update(args)
    max_depth = args["max_depth"]
    num_rounds = args["num_rounds"]
    n_feature = data[0].shape[1]
    train_size = data[0].shape[0]
    model_name = f"xgb_{max_depth}_{num_rounds}_{n_feature}_{train_size}.model"
    model_path = os.path.join(tmpdir, model_name)
    bst = xgb.train(params, dtrain, num_rounds)
    bst.save_model(model_path)

    return m.load(model_path, algo=args["fil_algo"],
                  output_class=args["output_class"],
                  threshold=args["threshold"],
                  storage_type=args["storage_type"])


def _build_fil_skl_classifier(m, data, args, tmpdir):
    """Trains an SKLearn classifier and returns a FIL version of it"""

    train_data, train_label = _training_data_to_numpy(data[0], data[1])

    params = {
        "n_estimators": 100,
        "max_leaf_nodes": 2**10,
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": 42,
    }
    params.update(args)

    # remove keyword arguments not understood by SKLearn
    for param_name in ["fil_algo", "output_class", "threshold",
                       "storage_type"]:
        params.pop(param_name, None)

    max_leaf_nodes = args["max_leaf_nodes"]
    n_estimators = args["n_estimators"]
    n_feature = data[0].shape[1]
    train_size = data[0].shape[0]
    model_name = (f"skl_{max_leaf_nodes}_{n_estimators}_{n_feature}_" +
                  f"{train_size}.model.pkl")
    model_path = os.path.join(tmpdir, model_name)
    skl_model = skl_ensemble.RandomForestClassifier(**params)
    skl_model.fit(train_data, train_label)
    pickle.dump(skl_model, open(model_path, "wb"))

    return m.load_from_sklearn(skl_model, algo=args["fil_algo"],
                               output_class=args["output_class"],
                               threshold=args["threshold"],
                               storage_type=args["storage_type"])


def _build_cpu_skl_classifier(m, data, args, tmpdir):
    """Loads the SKLearn classifier and returns it"""

    max_leaf_nodes = args["max_leaf_nodes"]
    n_estimators = args["n_estimators"]
    n_feature = data[0].shape[1]
    train_size = data[0].shape[0]
    model_name = (f"skl_{max_leaf_nodes}_{n_estimators}_{n_feature}_" +
                  f"{train_size}.model.pkl")
    model_path = os.path.join(tmpdir, model_name)

    skl_model = pickle.load(open(model_path, "rb"))
    return skl_model


def _build_treelite_classifier(m, data, args, tmpdir):
    """Setup function for treelite classification benchmarking"""
    from cuml.common.import_utils import has_xgboost
    import treelite
    import treelite_runtime
    if has_xgboost():
        import xgboost as xgb
    else:
        raise ImportError("No XGBoost package found")

    max_depth = args["max_depth"]
    num_rounds = args["num_rounds"]
    n_feature = data[0].shape[1]
    train_size = data[0].shape[0]
    model_name = f"xgb_{max_depth}_{num_rounds}_{n_feature}_{train_size}.model"
    model_path = os.path.join(tmpdir, model_name)

    bst = xgb.Booster()
    bst.load_model(model_path)
    tl_model = treelite.Model.from_xgboost(bst)
    tl_model.export_lib(
        toolchain="gcc", libpath=os.path.join(tmpdir, 'treelite.so'),
        params={'parallel_comp': 40}, verbose=False
    )
    return treelite_runtime.Predictor(os.path.join(tmpdir, 'treelite.so'),
                                      verbose=False)


def _treelite_fil_accuracy_score(y_true, y_pred):
    """Function to get correct accuracy for FIL (returns class index)"""
    # convert the input if necessary
    y_pred1 = (y_pred.copy_to_host() if
               cuda.devicearray.is_cuda_ndarray(y_pred) else y_pred)
    y_true1 = (y_true.copy_to_host() if
               cuda.devicearray.is_cuda_ndarray(y_true) else y_true)

    y_pred_binary = input_utils.convert_dtype(y_pred1 > 0.5, np.int32)
    return cuml.metrics.accuracy_score(y_true1, y_pred_binary)


def _build_mnmg_umap(m, data, args, tmpdir):
    client = args['client']
    del args['client']
    local_model = UMAP(**args)

    if isinstance(data, (tuple, list)):
        local_data = [x.compute() for x in data if x is not None]
    if len(local_data) == 2:
        X, y = local_data
        local_model.fit(X, y)
    else:
        X = local_data
        local_model.fit(X)

    return m(client=client, model=local_model, **args)

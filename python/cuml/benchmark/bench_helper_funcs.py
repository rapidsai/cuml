#
# Copyright (c) 2019, NVIDIA CORPORATION.
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
from cuml.utils import input_utils
import numpy as np
import pandas as pd
import cudf
from numba import cuda
from cuml.benchmark import datagen


def fit_kneighbors(m, x):
    m.fit(x)
    m.kneighbors(x)


def fit(m, x, y=None):
    m.fit(x) if y is None else m.fit(x, y)


def fit_transform(m, x):
    m.fit_transform(x)


def predict(m, x):
    m.predict(x)


def _training_data_to_numpy(X, y):
    """Convert input training data into numpy format"""
    if isinstance(X, np.ndarray):
        X_np = X
        y_np = y
    elif isinstance(X, cudf.DataFrame):
        X_np = X.as_gpu_matrix().copy_to_host()
        y_np = y.to_gpu_array().copy_to_host()
    elif cuda.devicearray.is_cuda_ndarray(X):
        X_np = X.copy_to_host()
        y_np = y.copy_to_host()
    elif isinstance(X, (pd.DataFrame, pd.Series)):
        X_np = datagen._convert_to_numpy(X)
        y_np = datagen._convert_to_numpy(y)
    else:
        raise TypeError("Received unsupported input type")
    return X_np, y_np


def _build_fil_classifier(m, data, arg={}, tmpdir=None):
    """Setup function for FIL classification benchmarking"""
    from cuml.utils.import_utils import has_xgboost
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
    params.update(arg)
    max_depth = arg["max_depth"]
    num_rounds = arg["num_rounds"]
    n_feature = data[0].shape[1]
    train_size = data[0].shape[0]
    model_name = f"xgb_{max_depth}_{num_rounds}_{n_feature}_{train_size}.model"
    model_path = os.path.join(tmpdir, model_name)
    bst = xgb.train(params, dtrain, num_rounds)
    bst.save_model(model_path)

    return m.load(model_path, algo=arg["fil_algo"],
                  output_class=arg["output_class"],
                  threshold=arg["threshold"],
                  storage_type=arg["storage_type"])


def _build_treelite_classifier(m, data, arg={}, tmpdir=None):
    """Setup function for treelite classification benchmarking"""
    from cuml.utils.import_utils import has_treelite, has_xgboost
    if has_treelite():
        import treelite
        import treelite.runtime
    else:
        raise ImportError("No treelite package found")
    if has_xgboost():
        import xgboost as xgb
    else:
        raise ImportError("No XGBoost package found")

    max_depth = arg["max_depth"]
    num_rounds = arg["num_rounds"]
    n_feature = data[0].shape[1]
    train_size = data[0].shape[0]
    model_name = f"xgb_{max_depth}_{num_rounds}_{n_feature}_{train_size}.model"
    model_path = os.path.join(tmpdir, model_name)

    bst = xgb.Booster()
    bst.load_model(model_path)
    tl_model = treelite.Model.from_xgboost(bst)
    tl_model.export_lib(
        toolchain="gcc", libpath=model_path+"treelite.so",
        params={'parallel_comp': 40}, verbose=False
    )
    return treelite.runtime.Predictor(model_path+"treelite.so", verbose=False)


def _treelite_fil_accuracy_score(y_true, y_pred):
    """Function to get correct accuracy for FIL (returns class index)"""
    y_pred_binary = input_utils.convert_dtype(y_pred > 0.5, np.int32)
    if isinstance(y_true, np.ndarray):
        return cuml.metrics.accuracy_score(y_true, y_pred_binary)
    elif cuda.devicearray.is_cuda_ndarray(y_true):
        y_true_np = y_true.copy_to_host()
        return cuml.metrics.accuracy_score(y_true_np, y_pred_binary)
    elif isinstance(y_true, cudf.Series):
        return cuml.metrics.accuracy_score(y_true, y_pred_binary)
    elif isinstance(y_true, pd.Series):
        return cuml.metrics.accuracy_score(y_true, y_pred_binary)
    else:
        raise TypeError("Received unsupported input type")

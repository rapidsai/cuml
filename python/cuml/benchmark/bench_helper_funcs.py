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
import tempfile
import cuml


def fit_kneighbors(m, x):
    m.fit(x)
    m.kneighbors(x)


def fit(m, x, y=None):
    m.fit(x) if y is None else m.fit(x, y)


def fit_transform(m, x):
    m.fit_transform(x)


def predict(m, x):
    m.predict(x)


def _build_fil_classifier(m, data, arg={}):
    """Setup function for FIL classification benchmarking"""
    from cuml.utils.import_utils import has_xgboost
    if has_xgboost():
        import xgboost as xgb
    else:
        raise ImportError("No XGBoost package found")

    # use maximum 1e5 rows to train the model
    train_size = min(data[0].shape[0], 100000)
    dtrain = xgb.DMatrix(data[0][:train_size, :], label=data[1][:train_size])
    params = {
        "silent": 1, "eval_metric": "error", "objective": "binary:logistic"
    }
    params.update(arg)
    max_depth = arg["max_depth"]
    num_rounds = arg["num_rounds"]
    n_feature = data[0].shape[1]

    tmpdir = tempfile.mkdtemp()
    model_name = f"xgb_{max_depth}_{num_rounds}_{n_feature}_{train_size}.model"
    model_path = os.path.join(tmpdir, model_name)

    bst = xgb.train(params, dtrain, num_rounds)
    bst.save_model(model_path)
    return m.load(model_path, algo=arg["fil_algo"],
                  output_class=arg["output_class"],
                  threshold=arg["threshold"],
                  storage_type=arg["storage_type"])


def _build_treelite_classifier(m, data, arg={}):
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

    # use maximum 1e5 rows to train the model
    train_size = min(data[0].shape[0], 100000)
    dtrain = xgb.DMatrix(data[0][:train_size, :], label=data[1][:train_size])
    params = {
        "silent": 1, "eval_metric": "error", "objective": "binary:logistic"
    }
    params.update(arg)
    max_depth = arg["max_depth"]
    num_rounds = arg["num_rounds"]
    n_feature = data[0].shape[1]

    tmpdir = tempfile.mkdtemp()
    model_name = f"xgb_{max_depth}_{num_rounds}_{n_feature}_{train_size}.model"
    model_path = os.path.join(tmpdir, model_name)

    bst = xgb.train(params, dtrain, num_rounds)
    tl_model = treelite.Model.from_xgboost(bst)
    tl_model.export_lib(
        toolchain="gcc", libpath=model_path+"treelite.so",
        params={'parallel_comp': 40}, verbose=False
    )
    return treelite.runtime.Predictor(model_path+"treelite.so", verbose=False)


def _treelite_fil_accuracy_score(y_true, y_pred):
    y_pred_binary = y_pred > 0.5
    return cuml.metrics.accuracy_score(y_true, y_pred_binary)

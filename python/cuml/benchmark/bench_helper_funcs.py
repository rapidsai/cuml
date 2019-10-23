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


def fit_kneighbors(m, x):
    m.fit(x)
    m.kneighbors(x)


def fit(m, x, y=None):
    m.fit(x) if y is None else m.fit(x, y)


def fit_transform(m, x):
    m.fit_transform(x)


def predict(m, x, y=None):
    m.predict(x) if y is None else m.predict(x, y)


def fil_classification_set_up(m, data, arg={}):
    from cuml.utils.import_utils import has_xgboost
    if has_xgboost():
        import xgboost as xgb
    else:
        raise ImportError("No XGBoost package found which is required for benchmarking FIL")
    import os 

    dtrain = xgb.DMatrix(data[0], label=data[1])
    params = {"silent": 1, "eval_metric": "error", "objective": "binary:logistic"}
    params.update(arg)
    tmp_path = "./"
    model_path = os.path.join(tmp_path, 'xgb_class.model')
    bst = xgb.train(params, dtrain, arg["num_rounds"])
    bst.save_model(model_path)

    obj = m.load(model_path, algo=arg["algo"], output_class=arg["output_class"], threshold=arg["threshold"])
    return obj
#
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
from cuml.manifold import UMAP
from cuml.benchmark import datagen
from cuml.common.device_selection import using_device_type
from cuml.internals.device_type import DeviceType
from cuml.internals.global_settings import GlobalSettings
from cuml.internals.safe_imports import (
    cpu_only_import,
    gpu_only_import,
    gpu_only_import_from,
    safe_import,
)
import sklearn.ensemble as skl_ensemble
import pickle as pickle
import os
import cuml
from cuml.internals import input_utils
from time import perf_counter

np = cpu_only_import("numpy")
pd = cpu_only_import("pandas")
cudf = gpu_only_import("cudf")
cuda = gpu_only_import_from("numba", "cuda")
cp = gpu_only_import("cupy")
xgb = safe_import("xgboost")
treelite = safe_import("treelite")


def call(m, func_name, X, y=None):
    def unwrap_and_get_args(func):
        if hasattr(func, "__wrapped__"):
            return unwrap_and_get_args(func.__wrapped__)
        else:
            return func.__code__.co_varnames

    if not hasattr(m, func_name):
        raise ValueError("Model does not have function " + func_name)
    func = getattr(m, func_name)
    argnames = unwrap_and_get_args(func)
    if y is not None and "y" in argnames:
        func(X, y=y)
    else:
        func(X)


def pass_func(m, x, y=None):
    pass


def fit(m, x, y=None):
    call(m, "fit", x, y)


def predict(m, x, y=None):
    call(m, "predict", x)


def transform(m, x, y=None):
    call(m, "transform", x)


def kneighbors(m, x, y=None):
    call(m, "kneighbors", x)


def fit_predict(m, x, y=None):
    if hasattr(m, "predict"):
        fit(m, x, y)
        predict(m, x)
    else:
        call(m, "fit_predict", x, y)


def fit_transform(m, x, y=None):
    if hasattr(m, "transform"):
        fit(m, x, y)
        transform(m, x)
    else:
        call(m, "fit_transform", x, y)


def fit_kneighbors(m, x, y=None):
    if hasattr(m, "kneighbors"):
        fit(m, x, y)
        kneighbors(m, x)
    else:
        call(m, "fit_kneighbors", x, y)


def _training_data_to_numpy(X, y):
    """Convert input training data into numpy format"""
    if isinstance(X, np.ndarray):
        X_np = X
        y_np = y
    elif isinstance(X, cp.ndarray):
        X_np = cp.asnumpy(X)
        y_np = cp.asnumpy(y)
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

    train_data, train_label = _training_data_to_numpy(data[0], data[1])

    dtrain = xgb.DMatrix(train_data, label=train_label)

    params = {
        "silent": 1,
        "eval_metric": "error",
        "objective": "binary:logistic",
        "tree_method": "gpu_hist",
    }
    params.update(args)
    max_depth = args["max_depth"]
    num_rounds = args["num_rounds"]
    n_feature = data[0].shape[1]
    train_size = data[0].shape[0]
    model_name = f"xgb_{max_depth}_{num_rounds}_{n_feature}_{train_size}.ubj"
    model_path = os.path.join(tmpdir, model_name)
    bst = xgb.train(params, dtrain, num_rounds)
    bst.save_model(model_path)

    fil_kwargs = {
        param: args[input_name]
        for param, input_name in (
            ("algo", "fil_algo"),
            ("output_class", "output_class"),
            ("threshold", "threshold"),
            ("storage_type", "storage_type"),
            ("precision", "precision"),
        )
        if input_name in args
    }

    return m.load(model_path, **fil_kwargs)


class OptimizedFilWrapper:
    """Helper class to make use of optimized parameters in both FIL and
    experimental FIL through a uniform interface"""

    def __init__(
        self, fil_model, optimal_chunk_size, experimental, infer_type="default"
    ):
        self.fil_model = fil_model
        self.predict_kwargs = {}
        if experimental:
            self.predict_kwargs["chunk_size"] = optimal_chunk_size
        self.infer_type = infer_type

    def predict(self, X):
        if self.infer_type == "per_tree":
            return self.fil_model.predict_per_tree(X, **self.predict_kwargs)
        return self.fil_model.predict(X, **self.predict_kwargs)


def _build_optimized_fil_classifier(m, data, args, tmpdir):
    """Setup function for FIL classification benchmarking with optimal
    parameters"""
    with using_device_type("gpu"):

        train_data, train_label = _training_data_to_numpy(data[0], data[1])

        dtrain = xgb.DMatrix(train_data, label=train_label)

        params = {
            "silent": 1,
            "eval_metric": "error",
            "objective": "binary:logistic",
            "tree_method": "gpu_hist",
        }
        params.update(args)
        max_depth = args["max_depth"]
        num_rounds = args["num_rounds"]
        n_feature = data[0].shape[1]
        train_size = data[0].shape[0]
        model_name = (
            f"xgb_{max_depth}_{num_rounds}_{n_feature}_{train_size}.model"
        )
        model_path = os.path.join(tmpdir, model_name)
        bst = xgb.train(params, dtrain, num_rounds)
        bst.save_model(model_path)

    allowed_chunk_sizes = [1, 2, 4, 8, 16, 32]
    if GlobalSettings().device_type is DeviceType.host:
        allowed_chunk_sizes.extend((64, 128, 256))

    fil_kwargs = {
        param: args[input_name]
        for param, input_name in (
            ("algo", "fil_algo"),
            ("output_class", "output_class"),
            ("threshold", "threshold"),
            ("storage_type", "storage_type"),
            ("precision", "precision"),
        )
        if input_name in args
    }
    experimental = m is cuml.experimental.ForestInference
    if experimental:
        allowed_storage_types = ["sparse"]
    else:
        allowed_storage_types = ["sparse", "sparse8"]
        if args["storage_type"] == "dense":
            allowed_storage_types.append("dense")
    infer_type = args.get("infer_type", "default")

    optimal_storage_type = "sparse"
    optimal_algo = "NAIVE"
    optimal_layout = "breadth_first"
    optimal_chunk_size = 1
    best_time = None
    optimization_cycles = 5
    for storage_type in allowed_storage_types:
        fil_kwargs["storage_type"] = storage_type
        allowed_algo_types = ["NAIVE"]
        if not experimental and storage_type == "dense":
            allowed_algo_types.extend(("TREE_REORG", "BATCH_TREE_REORG"))
        allowed_layout_types = ["breadth_first"]
        if experimental:
            allowed_layout_types.append("depth_first")
        for algo in allowed_algo_types:
            fil_kwargs["algo"] = algo
            for layout in allowed_layout_types:
                if experimental:
                    fil_kwargs["layout"] = layout
                for chunk_size in allowed_chunk_sizes:
                    fil_kwargs["threads_per_tree"] = chunk_size
                    call_args = {}
                    if experimental:
                        call_args = {"chunk_size": chunk_size}
                    fil_model = m.load(model_path, **fil_kwargs)
                    if infer_type == "per_tree":
                        fil_model.predict_per_tree(train_data, **call_args)
                    else:
                        fil_model.predict(train_data, **call_args)
                    begin = perf_counter()
                    if infer_type == "per_tree":
                        fil_model.predict_per_tree(train_data, **call_args)
                    else:
                        for _ in range(optimization_cycles):
                            fil_model.predict(train_data, **call_args)
                    end = perf_counter()
                    elapsed = end - begin
                    if best_time is None or elapsed < best_time:
                        best_time = elapsed
                        optimal_storage_type = storage_type
                        optimal_algo = algo
                        optimal_chunk_size = chunk_size
                        optimal_layout = layout

        fil_kwargs["storage_type"] = optimal_storage_type
        fil_kwargs["algo"] = optimal_algo
        fil_kwargs["threads_per_tree"] = optimal_chunk_size
        if experimental:
            fil_kwargs["layout"] = optimal_layout

        return OptimizedFilWrapper(
            m.load(model_path, **fil_kwargs),
            optimal_chunk_size,
            experimental,
            infer_type=infer_type,
        )


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
    for param_name in [
        "fil_algo",
        "output_class",
        "threshold",
        "storage_type",
        "precision",
    ]:
        params.pop(param_name, None)

    max_leaf_nodes = args["max_leaf_nodes"]
    n_estimators = args["n_estimators"]
    n_feature = data[0].shape[1]
    train_size = data[0].shape[0]
    model_name = (
        f"skl_{max_leaf_nodes}_{n_estimators}_{n_feature}_"
        + f"{train_size}.model.pkl"
    )
    model_path = os.path.join(tmpdir, model_name)
    skl_model = skl_ensemble.RandomForestClassifier(**params)
    skl_model.fit(train_data, train_label)
    pickle.dump(skl_model, open(model_path, "wb"))

    fil_kwargs = {
        param: args[input_name]
        for param, input_name in (
            ("algo", "fil_algo"),
            ("output_class", "output_class"),
            ("threshold", "threshold"),
            ("storage_type", "storage_type"),
            ("precision", "precision"),
        )
        if input_name in args
    }

    return m.load_from_sklearn(skl_model, **fil_kwargs)


def _build_cpu_skl_classifier(m, data, args, tmpdir):
    """Loads the SKLearn classifier and returns it"""

    max_leaf_nodes = args["max_leaf_nodes"]
    n_estimators = args["n_estimators"]
    n_feature = data[0].shape[1]
    train_size = data[0].shape[0]
    model_name = (
        f"skl_{max_leaf_nodes}_{n_estimators}_{n_feature}_"
        + f"{train_size}.model.pkl"
    )
    model_path = os.path.join(tmpdir, model_name)

    skl_model = pickle.load(open(model_path, "rb"))
    return skl_model


class GtilWrapper:
    """Helper class to provide interface to GTIL compatible with
    benchmarking functions"""

    def __init__(self, tl_model, infer_type="default"):
        self.tl_model = tl_model
        self.infer_type = infer_type

    def predict(self, X):
        if self.infer_type == "per_tree":
            return treelite.gtil.predict_per_tree(self.tl_model, X)
        return treelite.gtil.predict(self.tl_model, X)


def _build_gtil_classifier(m, data, args, tmpdir):
    """Setup function for treelite classification benchmarking"""

    max_depth = args["max_depth"]
    num_rounds = args["num_rounds"]
    infer_type = args.get("infer_type", "default")
    n_feature = data[0].shape[1]
    train_size = data[0].shape[0]
    model_name = f"xgb_{max_depth}_{num_rounds}_{n_feature}_{train_size}.model"
    model_path = os.path.join(tmpdir, model_name)

    bst = xgb.Booster()
    bst.load_model(model_path)
    tl_model = treelite.Model.from_xgboost(bst)
    return GtilWrapper(tl_model, infer_type=infer_type)


def _treelite_fil_accuracy_score(y_true, y_pred):
    """Function to get correct accuracy for FIL (returns class index)"""
    # convert the input if necessary
    y_pred1 = (
        y_pred.copy_to_host()
        if cuda.devicearray.is_cuda_ndarray(y_pred)
        else y_pred
    )
    y_true1 = (
        y_true.copy_to_host()
        if cuda.devicearray.is_cuda_ndarray(y_true)
        else y_true
    )

    y_pred_binary = input_utils.convert_dtype(y_pred1 > 0.5, np.int32)
    return cuml.metrics.accuracy_score(y_true1, y_pred_binary)


def _build_mnmg_umap(m, data, args, tmpdir):
    client = args["client"]
    del args["client"]
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

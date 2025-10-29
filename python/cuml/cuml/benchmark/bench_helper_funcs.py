#
# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import os
import pickle as pickle
from time import perf_counter

import cudf
import cupy as cp
import numpy as np
import pandas as pd
import sklearn.ensemble as skl_ensemble
from numba import cuda

import cuml
from cuml.benchmark import datagen
from cuml.fil import get_fil_device_type, set_fil_device_type
from cuml.internals import input_utils
from cuml.internals.device_type import DeviceType
from cuml.manifold import UMAP

try:
    import xgboost as xgb
except ImportError:
    xgb = None


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


def train_xgboost(m, x, y=None):
    """
    Bench function for XGBoost that times the training phase.
    
    This function is designed to work with XGBoostTrainWrapper instances
    that have been pre-configured by the setup functions. It calls the
    retrain() method which performs the actual training that gets timed.
    
    Parameters
    ----------
    m : XGBoostTrainWrapper
    x : array-like
    y : array-like, optional
    
    Returns
    -------
    booster : xgboost.Booster or None
        The trained XGBoost booster model
    
    Notes
    -----
    The x and y parameters are present for interface consistency with other
    benchmark functions but are not used since the training data is already
    contained in the pre-built DMatrix within the wrapper. This is because
    The benchmarking framework always calls bench_func with 
    (model, data[0], data[1]) or (model, data[0]).
    """
    if hasattr(m, 'retrain'):
        return m.retrain()
    else:
        raise ValueError(
            f"Expected XGBoostTrainWrapper with 'retrain' method, "
            f"but got {type(m).__name__}. Ensure the setup function "
            f"(_build_xgboost_*_for_training) was called correctly."
        )


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
    import xgboost as xgb

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
            ("is_classifier", "is_classifier"),
            ("threshold", "threshold"),
            ("precision", "precision"),
            ("layout", "layout"),
        )
        if input_name in args
    }

    return m.load(model_path, **fil_kwargs)


class OptimizedFilWrapper:
    """Helper class to make use of optimized parameters in FIL"""

    def __init__(self, fil_model, optimal_chunk_size, infer_type="default"):
        self.fil_model = fil_model
        self.predict_kwargs = {"chunk_size": optimal_chunk_size}
        self.infer_type = infer_type

    def predict(self, X):
        if self.infer_type == "per_tree":
            return self.fil_model.predict_per_tree(X, **self.predict_kwargs)
        return self.fil_model.predict(X, **self.predict_kwargs)


def _build_optimized_fil_classifier(m, data, args, tmpdir):
    """Setup function for FIL classification benchmarking with optimal
    parameters"""
    import xgboost as xgb

    with set_fil_device_type("gpu"):

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
            f"xgb_{max_depth}_{num_rounds}_{n_feature}_{train_size}.ubj"
        )
        model_path = os.path.join(tmpdir, model_name)
        bst = xgb.train(params, dtrain, num_rounds)
        bst.save_model(model_path)

    allowed_chunk_sizes = [1, 2, 4, 8, 16, 32]
    if get_fil_device_type() is DeviceType.host:
        allowed_chunk_sizes.extend((64, 128, 256))

    fil_kwargs = {
        param: args[input_name]
        for param, input_name in (
            ("is_classifier", "is_classifier"),
            ("threshold", "threshold"),
            ("precision", "precision"),
            ("layout", "layout"),
        )
        if input_name in args
    }
    infer_type = args.get("infer_type", "default")

    optimal_layout = "breadth_first"
    optimal_chunk_size = 1
    best_time = None
    optimization_cycles = 5

    allowed_layout_types = ["breadth_first", "depth_first", "layered"]
    for layout in allowed_layout_types:
        fil_kwargs["layout"] = layout
        for chunk_size in allowed_chunk_sizes:
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
                optimal_chunk_size = chunk_size
                optimal_layout = layout

        fil_kwargs["layout"] = optimal_layout

        return OptimizedFilWrapper(
            m.load(model_path, **fil_kwargs),
            optimal_chunk_size,
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
        "is_classifier",
        "threshold",
        "precision",
        "layout",
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
            ("is_classifier", "is_classifier"),
            ("threshold", "threshold"),
            ("precision", "precision"),
            ("layout", "layout"),
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
        import treelite

        if self.infer_type == "per_tree":
            return treelite.gtil.predict_per_tree(self.tl_model, X)
        return treelite.gtil.predict(self.tl_model, X)


def _build_gtil_classifier(m, data, args, tmpdir):
    """Setup function for treelite classification benchmarking"""
    import treelite
    import xgboost as xgb

    max_depth = args["max_depth"]
    num_rounds = args["num_rounds"]
    infer_type = args.get("infer_type", "default")
    n_feature = data[0].shape[1]
    train_size = data[0].shape[0]
    model_name = f"xgb_{max_depth}_{num_rounds}_{n_feature}_{train_size}.ubj"
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


class XGBoostTrainWrapper:
    """Helper class for benchmarking XGBoost training phase"""

    def __init__(
        self,
        dtrain,
        params,
        num_boost_round,
        device="cpu",
    ):
        self.dtrain = dtrain
        self.params = params
        self.num_boost_round = num_boost_round
        self.device = device
        self.booster = None

    def retrain(self):
        """Retrain the model - this is what gets timed"""

        debug_mode = os.environ.get("XGBOOST_DEBUG", "0") == "1"
        if debug_mode:
            print(f"[XGBoost Retrain] Device: {self.device}, Rounds: {self.num_boost_round}")

        self.booster = xgb.train(self.params, self.dtrain, self.num_boost_round)
        return self.booster


def _build_xgboost_for_training(m, data, args, tmpdir, task_type="classification"):
    """
    Common setup function for XGBoost training - prepares but doesn't train (for timing training).
    
    Args:
        m: Model (unused but required for interface consistency)
        data: Training data tuple (features, labels)
        args: Configuration arguments
        tmpdir: Temporary directory (unused)
        task_type: Either "classification" or "regression"
    
    Returns:
        XGBoostTrainWrapper ready for training
    """

    train_data, train_label = _training_data_to_numpy(data[0], data[1])

    args_copy = args.copy()

    use_quantile_dmatrix = args_copy.pop("use_quantile_dmatrix", False)
    max_bin = args_copy.pop("max_bin", 256)
    num_boost_round = args_copy.pop("n_estimators", 100)
    device = args_copy.pop("device", "cpu")
    
    debug_mode = os.environ.get("XGBOOST_DEBUG", "0") == "1"
    if debug_mode:
        task_name = "Classifier" if task_type == "classification" else "Regressor"
        print(f"[XGBoost Setup {task_name} for Training] Device: {device}")

    # Task-specific label processing and parameter setup
    if task_type == "classification":
        unique_labels = np.unique(train_label)
        n_classes = len(unique_labels)
        
        label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        train_label_normalized = np.array([label_map[label] for label in train_label])
        
        # Determine objective based on number of classes
        if n_classes == 2:
            objective = "binary:logistic"
            eval_metric = "error"
        else:
            objective = "multi:softmax"
            eval_metric = "merror"
        
        params = {
            "objective": objective,
            "eval_metric": eval_metric,
            "device": device,
        }
        
        if n_classes > 2:
            params["num_class"] = n_classes
    else:
        train_label_normalized = train_label
        
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "device": device,
        }

    if use_quantile_dmatrix:
        dtrain = xgb.QuantileDMatrix(
            train_data, label=train_label_normalized, max_bin=max_bin
        )
    else:
        dtrain = xgb.DMatrix(train_data, label=train_label_normalized)

    params.update(args_copy)
    
    if debug_mode:
        task_name = "Classifier" if task_type == "classification" else "Regressor"
        print(f"[XGBoost Setup {task_name} for Training] Final params: {params}")

    # Return wrapper that can retrain (training will be timed)
    return XGBoostTrainWrapper(
        dtrain,
        params,
        num_boost_round,
        device=device,
    )


def _build_xgboost_classifier_for_training(m, data, args, tmpdir):
    """Setup function for XGBoost classification - prepares but doesn't train (for timing training)"""
    return _build_xgboost_for_training(m, data, args, tmpdir, task_type="classification")


def _build_xgboost_regressor_for_training(m, data, args, tmpdir):
    """Setup function for XGBoost regression - prepares but doesn't train (for timing training)"""
    return _build_xgboost_for_training(m, data, args, tmpdir, task_type="regression")

#
# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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
from __future__ import annotations

import math
import typing
import warnings
from typing import Literal

import cupy as cp
import numpy as np
import treelite.sklearn
from pylibraft.common.handle import Handle

import cuml.internals
from cuml.common import input_to_cuml_array
from cuml.common.exceptions import NotFittedError
from cuml.fil.fil import ForestInference
from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from cuml.internals.interop import (
    InteropMixin,
    UnsupportedOnCPU,
    UnsupportedOnGPU,
)
from cuml.internals.treelite import safe_treelite_call
from cuml.prims.label.classlabels import check_labels, make_monotonic

from cuml.ensemble.randomforest_shared cimport *
from cuml.internals.treelite cimport *

from cuda.bindings import runtime

_split_criterion_lookup = {
    "0": GINI,
    "gini": GINI,
    "1": ENTROPY,
    "entropy": ENTROPY,
    "2": MSE,
    "mse": MSE,
    "3": MAE,
    "mae": MAE,
    "4": POISSON,
    "poisson": POISSON,
    "5": GAMMA,
    "gamma": GAMMA,
    "6": INVERSE_GAUSSIAN,
    "inverse_gaussian": INVERSE_GAUSSIAN,
    "7": CRITERION_END
}

_criterion_to_split_criterion = {
    "gini": "gini",
    "entropy": "entropy",
    "poisson": "poisson",
    "squared_error": "mse",
}

_split_criterion_to_criterion = {
    GINI: "gini",
    ENTROPY: "entropy",
    POISSON: "poisson",
    MSE: "squared_error",
}


def _normalize_split_criterion(split_criterion):
    if (out := _split_criterion_lookup.get(str(split_criterion))) is None:
        warnings.warn(
            "The split criterion chosen was not present in the list of options accepted "
            "by the model and so the CRITERION_END option has been chosen."
        )
        return CRITERION_END
    if out == MAE:
        raise NotImplementedError(
            "cuML does not currently support mean average error as a"
            " RandomForest split criterion"
        )
    return out


def compute_max_features(
    max_features: Literal["sqrt", "log2", None] | int | float,
    n_cols: int,
) -> float:
    if isinstance(max_features, int):
        return max_features / n_cols
    elif isinstance(max_features, float):
        return max_features
    elif max_features == 'sqrt':
        return math.sqrt(n_cols) / n_cols
    elif max_features == 'log2':
        return math.log2(n_cols) / n_cols
    elif max_features is None:
        return 1.0
    else:
        raise ValueError(
            f"Expected `max_features` to be an int, float, None, or one of "
            f"['sqrt', 'log2']. Got {max_features!r} instead."
        )


class BaseRandomForestModel(Base, InteropMixin):

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "split_criterion",
            "n_estimators",
            "bootstrap",
            "max_samples",
            "max_depth",
            "max_leaves",
            "max_features",
            "n_bins",
            "min_samples_leaf",
            "min_samples_split",
            "min_impurity_decrease",
            "max_batch_size",
            "random_state",
            "criterion",
            "n_streams",
            "handle",
            "verbose",
            "output_type",
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        if model.oob_score:
            raise UnsupportedOnGPU("`oob_score=True` is not supported")

        if model.warm_start:
            raise UnsupportedOnGPU("`warm_start=True` is not supported")

        if model.monotonic_cst is not None:
            raise UnsupportedOnGPU(f"`monotonic_cst={model.monotonic_cst!r} is not supported")

        if model.ccp_alpha != 0:
            raise UnsupportedOnGPU(f"`ccp_alpha={model.ccp_alpha}` is not supported")

        if model.min_weight_fraction_leaf != 0:
            raise UnsupportedOnGPU(
                f"`min_weight_fraction_leaf={model.min_weight_fraction_leaf}` is not supported"
            )

        if (split_criterion := _criterion_to_split_criterion.get(model.criterion)) is None:
            raise UnsupportedOnGPU(f"`criterion={model.criterion!r}` is not supported")

        # We only forward some parameters, falling back to cuml defaults otherwise
        conditional_params = {}

        if isinstance(model.max_samples, int):
            raise UnsupportedOnGPU("`int` values for `max_samples` are not supported")
        elif model.max_samples is not None:
            conditional_params["max_samples"] = model.max_samples

        if model.random_state is not None:
            # determinism requires 1 CUDA stream
            conditional_params["n_streams"] = 1

        if model.max_depth is not None:
            conditional_params["max_depth"] = model.max_depth

        return {
            "n_estimators": model.n_estimators,
            "split_criterion": split_criterion,
            "min_samples_split": model.min_samples_split,
            "min_samples_leaf": model.min_samples_leaf,
            "max_features": model.max_features,
            "max_leaves": -1 if model.max_leaf_nodes is None else model.max_leaf_nodes,
            "min_impurity_decrease": model.min_impurity_decrease,
            "bootstrap": model.bootstrap,
            "random_state": model.random_state,
            **conditional_params
        }

    def _params_to_cpu(self):
        if (criterion := _split_criterion_to_criterion.get(self.split_criterion)) is None:
            raise UnsupportedOnCPU(
                f"`split_criterion={self.split_criterion!r}` is not supported"
            )

        return {
            "n_estimators": self.n_estimators,
            "criterion": criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "max_leaf_nodes": None if self.max_leaves == -1 else self.max_leaves,
            "min_impurity_decrease": self.min_impurity_decrease,
            "bootstrap": self.bootstrap,
            "random_state": self.random_state,
            "max_samples": self.max_samples,
        }

    def _attrs_from_cpu(self, model):
        tl_model = treelite.sklearn.import_model(model)
        return {
            "treelite_serialized_bytes": tl_model.serialize_bytes(),
            "dtype": np.float64,
            "update_labels": False,
            "n_outputs_": model.n_outputs_,
            "n_rows": model._n_samples,
            "n_cols": model.n_features_in_,
            **super()._attrs_from_cpu(model)
        }

    def _attrs_to_cpu(self, model):
        self._serialize_treelite_bytes()
        tl_model = treelite.Model.deserialize_bytes(self.treelite_serialized_bytes)
        sk_model = treelite.sklearn.export_model(tl_model)

        # Compute _n_samples_bootstrap
        if self.max_samples is None:
            n_samples_bootstrap = self.n_rows
        else:
            n_samples_bootstrap = max(round(self.n_rows * self.max_samples), 1)

        return {
            "estimator_": sk_model.estimator,
            "estimators_": sk_model.estimators_,
            "n_outputs_": self.n_outputs_,
            "_n_samples": self.n_rows,
            "_n_samples_bootstrap": n_samples_bootstrap,
            **super()._attrs_to_cpu(model)
        }

    def __init__(
        self,
        *,
        split_criterion,
        n_estimators=100,
        bootstrap=True,
        max_samples=1.0,
        max_depth=16,
        max_leaves=-1,
        max_features='sqrt',
        n_bins=128,
        min_samples_leaf=1,
        min_samples_split=2,
        min_impurity_decrease=0.0,
        max_batch_size=4096,
        random_state=None,
        criterion=None,
        n_streams=4,
        handle=None,
        verbose=False,
        output_type=None,
    ):
        if handle is None:
            handle = Handle(n_streams=n_streams)

        super().__init__(handle=handle, verbose=verbose, output_type=output_type)

        if max_depth <= 0:
            raise ValueError("Must specify max_depth >0 ")

        self.split_criterion = _normalize_split_criterion(split_criterion)
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.max_features = max_features
        self.n_bins = n_bins
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_batch_size = max_batch_size
        self.random_state = random_state
        self.n_streams = n_streams

        self.rf_forest = 0
        self.rf_forest64 = 0
        self.treelite_serialized_bytes = None
        self.n_cols = None

    def set_params(self, **params):
        self.treelite_serialized_bytes = None
        return super().set_params(**params)

    def __len__(self):
        """Return the number of estimators in the ensemble."""
        return self.n_estimators

    def _serialize_treelite_bytes(self) -> bytes:
        """
        Serialize the cuML RF model as bytes.
        Internally, the RF model is serialized as follows:
        * The RF model is converted to a Treelite model object.
        * The Treelite model object is then serialized as bytes.

        The serialized byte sequence will be internally cached to
        self.treelite_serialized_bytes.
        """
        if self.treelite_serialized_bytes:  # Cache hit
            return self.treelite_serialized_bytes

        if not self.rf_forest:
            raise NotFittedError("Attempting to serialize an un-fit forest.")

        # Convert RF model object to Treelite model object
        if self.dtype not in [np.float32, np.float64]:
            raise ValueError("Unknown dtype.")

        cdef TreeliteModelHandle tl_handle = NULL
        if self.RF_type == CLASSIFICATION:
            if self.dtype == np.float32:
                build_treelite_forest(
                    &tl_handle,
                    <RandomForestMetaData[float, int]*>
                    <uintptr_t> self.rf_forest,
                    <int> self.n_cols
                )
            elif self.dtype == np.float64:
                build_treelite_forest(
                    &tl_handle,
                    <RandomForestMetaData[double, int]*>
                    <uintptr_t> self.rf_forest64,
                    <int> self.n_cols
                )
        else:
            if self.dtype == np.float32:
                build_treelite_forest(
                    &tl_handle,
                    <RandomForestMetaData[float, float]*>
                    <uintptr_t> self.rf_forest,
                    <int> self.n_cols
                )
            elif self.dtype == np.float64:
                build_treelite_forest(
                    &tl_handle,
                    <RandomForestMetaData[double, double]*>
                    <uintptr_t> self.rf_forest64,
                    <int> self.n_cols
                )

        cdef const char* tl_bytes = NULL
        cdef size_t tl_bytes_len
        safe_treelite_call(
            TreeliteSerializeModelToBytes(tl_handle, &tl_bytes, &tl_bytes_len),
            "Failed to serialize RF model to bytes due to internal error: "
            "Failed to serialize Treelite model to bytes."
        )
        cdef bytes tl_serialized_bytes = tl_bytes[:tl_bytes_len]
        self.treelite_serialized_bytes = tl_serialized_bytes
        return self.treelite_serialized_bytes

    def _deserialize_from_treelite(self, tl_model):
        """
        Update the cuML RF model to match the given Treelite model.
        """
        self._reset_forest_data()
        self.treelite_serialized_bytes = tl_model.serialize_bytes()
        self.n_cols = tl_model.num_feature
        self.n_estimators = tl_model.num_tree

        return self

    @cuml.internals.api_base_return_generic(set_output_type=True,
                                            set_n_features_in=True,
                                            get_output_type=False)
    def _dataset_setup_for_fit(
            self, X, y,
            convert_dtype) -> typing.Tuple[CumlArray, CumlArray, float]:
        # Reset the old tree data for new fit call
        self._reset_forest_data()

        X_m, self.n_rows, self.n_cols, self.dtype = \
            input_to_cuml_array(X,
                                convert_to_dtype=(np.float32 if convert_dtype
                                                  else None),
                                check_dtype=[np.float32, np.float64],
                                order='F')
        if self.n_bins > self.n_rows:
            warnings.warn("The number of bins, `n_bins` is greater than "
                          "the number of samples used for training. "
                          "Changing `n_bins` to number of training samples.")
            self.n_bins = self.n_rows

        if self.RF_type == CLASSIFICATION:
            y_m, _, _, y_dtype = \
                input_to_cuml_array(
                    y, check_dtype=np.int32,
                    convert_to_dtype=(np.int32 if convert_dtype
                                      else None),
                    check_rows=self.n_rows, check_cols=1)
            if y_dtype != np.int32:
                raise TypeError("The labels `y` need to be of dtype"
                                " `int32`")
            self.classes_ = cp.unique(y_m)
            self.num_classes = self.n_classes_ = len(self.classes_)
            self.use_monotonic = not check_labels(
                y_m, cp.arange(self.num_classes, dtype=np.int32))
            if self.use_monotonic:
                y_m, _ = make_monotonic(y_m)

        else:
            y_m, _, _, y_dtype = \
                input_to_cuml_array(
                    y,
                    convert_to_dtype=(self.dtype if convert_dtype
                                      else None),
                    check_rows=self.n_rows, check_cols=1)

        if len(y_m.shape) == 1:
            self.n_outputs_ = 1
        else:
            self.n_outputs_ = y_m.shape[1]
        self.n_features_in_ = X_m.shape[1]

        max_feature_val = compute_max_features(self.max_features, self.n_cols)
        if isinstance(self.min_samples_leaf, float):
            self.min_samples_leaf = \
                math.ceil(self.min_samples_leaf * self.n_rows)
        if isinstance(self.min_samples_split, float):
            self.min_samples_split = \
                max(2, math.ceil(self.min_samples_split * self.n_rows))
        return X_m, y_m, max_feature_val

    def _predict_model_on_gpu(
        self,
        X,
        is_classifier = False,
        predict_proba = False,
        threshold = 0.5,
        convert_dtype = True,
        layout = "depth_first",
        default_chunk_size = None,
        align_bytes = None,
    ) -> CumlArray:
        treelite_bytes = self._serialize_treelite_bytes()
        status, current_device_id = runtime.cudaGetDevice()
        if status != runtime.cudaError_t.cudaSuccess:
            _, name = runtime.cudaGetErrorName(status)
            _, msg = runtime.cudaGetErrorString(status)
            raise RuntimeError(f"Failed to run cudaGetDevice(). {name}: {msg}")
        fil_model = ForestInference(
            treelite_model=treelite_bytes,
            handle=self.handle,
            output_type=self.output_type,
            verbose=self.verbose,
            is_classifier=is_classifier,
            layout=layout,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
            device_id=current_device_id,
        )
        if predict_proba:
            return fil_model.predict_proba(X)
        return fil_model.predict(X, threshold=threshold)

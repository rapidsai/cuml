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
import warnings
from typing import Literal

import numpy as np
import treelite.sklearn
from pylibraft.common.handle import Handle

from cuml.fil.fil import ForestInference
from cuml.internals.base import Base
from cuml.internals.interop import (
    InteropMixin,
    UnsupportedOnCPU,
    UnsupportedOnGPU,
)
from cuml.internals.treelite import safe_treelite_call
from cuml.internals.utils import check_random_seed

from libc.stdint cimport uint64_t, uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t

from cuml.internals.logger cimport level_enum
from cuml.internals.treelite cimport (
    TreeliteFreeModel,
    TreeliteModelHandle,
    TreeliteSerializeModelToBytes,
)


cdef extern from "cuml/ensemble/randomforest.hpp" namespace "ML" nogil:
    cdef enum CRITERION:
        GINI,
        ENTROPY,
        MSE,
        MAE,
        POISSON,
        GAMMA,
        INVERSE_GAUSSIAN,
        CRITERION_END

    cdef struct RF_params:
        pass

    cdef RF_params set_rf_params(
        int max_depth,
        int max_leaves,
        float max_features,
        int max_n_bins,
        int min_samples_leaf,
        int min_samples_split,
        float min_impurity_decrease,
        bool bootstrap,
        int n_trees,
        float max_samples,
        uint64_t seed,
        CRITERION split_criterion,
        int cfg_n_streams,
        int max_batch_size
    ) except +

    cdef void fit_treelite[T, L](
        handle_t& handle,
        TreeliteModelHandle* model,
        T* values,
        int n_rows,
        int n_cols,
        L* labels,
        int n_unique_labels,
        RF_params params,
        level_enum verbosity
    ) except +

    cdef void fit_treelite[T, L](
        handle_t& handle,
        TreeliteModelHandle* model,
        T* values,
        int n_rows,
        int n_cols,
        L* labels,
        RF_params params,
        level_enum verbosity
    ) except +


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
    "squared_error": "mse",
    "poisson": "poisson",
}

_split_criterion_to_criterion = {
    "gini": "gini",
    0: "gini",
    "entropy": "entropy",
    1: "entropy",
    "mse": "squared_error",
    2: "squared_error",
    "poisson": "poisson",
    4: "poisson",
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
            "_treelite_model_bytes": tl_model.serialize_bytes(),
            "n_outputs_": model.n_outputs_,
            "_n_samples": model._n_samples,
            "_n_samples_bootstrap": model._n_samples_bootstrap,
            **super()._attrs_from_cpu(model)
        }

    def _attrs_to_cpu(self, model):
        tl_model = treelite.Model.deserialize_bytes(self._treelite_model_bytes)
        sk_model = treelite.sklearn.export_model(tl_model)
        return {
            "estimator_": sk_model.estimator,
            "estimators_": sk_model.estimators_,
            "n_outputs_": self.n_outputs_,
            "_n_samples": self._n_samples,
            "_n_samples_bootstrap": self._n_samples_bootstrap,
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

        self.split_criterion = split_criterion
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

    def __getstate__(self):
        state = self.__dict__.copy()
        # FIL model isn't currently pickleable
        state.pop("_fil_model", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __len__(self):
        """Return the number of estimators in the ensemble."""
        return self.n_estimators

    def as_treelite(self):
        """
        Converts this estimator to a Treelite model.

        Returns
        -------
        treelite.Model
        """
        return treelite.Model.deserialize_bytes(self._treelite_model_bytes)

    def as_fil(
        self, layout="depth_first", default_chunk_size=None, align_bytes=None,
    ):
        """
        Create a Forest Inference (FIL) model from the trained cuML
        Random Forest model.

        Parameters
        ----------
        layout : string (default = 'depth_first')
            Specifies the in-memory layout of nodes in FIL forests. Options:
            'depth_first', 'layered', 'breadth_first'.
        default_chunk_size : int, optional (default = None)
            Determines how batches are further subdivided for parallel processing.
            The optimal value depends on hardware, model, and batch size.
            If None, will be automatically determined.
        align_bytes : int, optional (default = None)
            If specified, trees will be padded such that their in-memory size is
            a multiple of this value. This can improve performance by guaranteeing
            that memory reads from trees begin on a cache line boundary.
            Typical values are 0 or 128 on GPU and 0 or 64 on CPU.

        Returns
        -------
        fil_model : ForestInference
            A Forest Inference model which can be used to perform
            inferencing on the random forest model.
        """
        return ForestInference(
            handle=self.handle,
            verbose=self.verbose,
            output_type=self.output_type,
            treelite_model=self._treelite_model_bytes,
            is_classifier=(self._estimator_type == "classifier"),
            layout=layout,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
        )

    def _fit_forest(self, X, y):
        cdef bool is_classifier = self._estimator_type == "classifier"
        cdef bool is_float32 = X.dtype == np.float32

        cdef uintptr_t X_ptr = X.ptr
        cdef uintptr_t y_ptr = y.ptr
        cdef int n_rows = X.shape[0]
        cdef int n_cols = X.shape[1]
        cdef level_enum verbose = <level_enum> self.verbose
        cdef int n_classes = self.n_classes_ if is_classifier else 0

        if self.max_depth <= 0:
            raise ValueError("Must specify max_depth > 0")

        cdef float max_features = compute_max_features(self.max_features, n_cols)
        cdef uint64_t seed = (
            0 if self.random_state is None
            else check_random_seed(self.random_state)
        )
        cdef int min_samples_leaf = (
            self.min_samples_leaf if isinstance(self.min_samples_leaf, int)
            else math.ceil(self.min_samples_leaf * n_rows)
        )
        cdef int min_samples_split = (
            self.min_samples_split if isinstance(self.min_samples_split, int)
            else max(2, math.ceil(self.min_samples_split * n_rows))
        )

        cdef int n_bins
        if self.n_bins > n_rows:
            warnings.warn("The number of bins, `n_bins` is greater than "
                          "the number of samples used for training. "
                          "Changing `n_bins` to number of training samples.")
            n_bins = n_rows
        else:
            n_bins = self.n_bins

        cdef RF_params params = set_rf_params(
            self.max_depth,
            self.max_leaves,
            max_features,
            n_bins,
            min_samples_leaf,
            min_samples_split,
            self.min_impurity_decrease,
            self.bootstrap,
            self.n_estimators,
            self.max_samples,
            seed,
            _normalize_split_criterion(self.split_criterion),
            self.n_streams,
            self.max_batch_size,
        )

        cdef TreeliteModelHandle tl_handle
        cdef handle_t* handle_ = <handle_t*><uintptr_t>self.handle.getHandle()

        with nogil:
            if is_classifier:
                if is_float32:
                    fit_treelite(
                        handle_[0],
                        &tl_handle,
                        <float*> X_ptr,
                        n_rows,
                        n_cols,
                        <int*> y_ptr,
                        n_classes,
                        params,
                        verbose
                    )
                else:
                    fit_treelite(
                        handle_[0],
                        &tl_handle,
                        <double*> X_ptr,
                        n_rows,
                        n_cols,
                        <int*> y_ptr,
                        n_classes,
                        params,
                        verbose
                    )
            else:
                if is_float32:
                    fit_treelite(
                        handle_[0],
                        &tl_handle,
                        <float*> X_ptr,
                        n_rows,
                        n_cols,
                        <float*> y_ptr,
                        params,
                        verbose
                    )
                else:
                    fit_treelite(
                        handle_[0],
                        &tl_handle,
                        <double*> X_ptr,
                        n_rows,
                        n_cols,
                        <double*> y_ptr,
                        params,
                        verbose
                    )

        # XXX: Theoretically we could wrap `tl_handle` with `treelite.Model` to
        # manage ownership, and keep the loaded model around. However, this
        # only works if the `libtreelite` is ABI compatible with the one used
        # by `cuml`. This is currently true for conda environments, but not for
        # wheels where `cuml` and `treelite` use different manylinux ABIs. So
        # for now we need to do this serialize-and-reload dance. If/when this
        # is fixed we could instead store the loaded model and use that instead.
        cdef const char* tl_bytes = NULL
        cdef size_t tl_bytes_len
        safe_treelite_call(
            TreeliteSerializeModelToBytes(tl_handle, &tl_bytes, &tl_bytes_len),
            "Failed to serialize Treelite model to bytes:"
        )
        safe_treelite_call(
            TreeliteFreeModel(tl_handle), "Failed to free Treelite model:"
        )

        self._n_samples = y.shape[0]
        self._n_samples_bootstrap = (
            self._n_samples if self.max_samples is None
            else max(round(self._n_samples * self.max_samples), 1)
        )
        self.n_outputs_ = 1
        self._treelite_model_bytes = <bytes>(tl_bytes[:tl_bytes_len])
        # Ensure cached fil model is reset
        self._fil_model = None
        return self

    def _get_inference_fil_model(
        self,
        layout="depth_first",
        default_chunk_size=None,
        align_bytes=None,
    ):
        if (
            layout == "depth_first" and default_chunk_size is None and align_bytes is None
        ):
            # default parameters, get (or create) the cached fil model
            if (fil_model := getattr(self, "_fil_model", None)) is None:
                fil_model = self._fil_model = self.as_fil()
        else:
            fil_model = self.as_fil(
                layout=layout,
                default_chunk_size=default_chunk_size,
                align_bytes=align_bytes,
            )
        return fil_model

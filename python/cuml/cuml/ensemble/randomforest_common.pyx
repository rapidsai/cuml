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
import math
import typing
import warnings

import cupy as cp
import numpy as np
import treelite.sklearn
from pylibraft.common.handle import Handle

import cuml.accel
import cuml.internals
from cuml.common.exceptions import NotFittedError
from cuml.fil.fil import ForestInference
from cuml.internals.api_decorators import device_interop_preparation
from cuml.internals.array import CumlArray
from cuml.internals.base import UniversalBase
from cuml.internals.treelite import safe_treelite_call

from cuml.ensemble.randomforest_shared cimport *
from cuml.internals.treelite cimport *

from cuml.common import input_to_cuml_array
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.prims.label.classlabels import check_labels, make_monotonic


class BaseRandomForestModel(UniversalBase):
    _param_names = ['n_estimators', 'max_depth', 'handle',
                    'max_features', 'n_bins',
                    'split_criterion', 'min_samples_leaf',
                    'min_samples_split',
                    'min_impurity_decrease',
                    'bootstrap',
                    'verbose', 'max_samples',
                    'max_leaves',
                    'accuracy_metric', 'max_batch_size',
                    'n_streams', 'dtype',
                    'output_type', 'min_weight_fraction_leaf', 'n_jobs',
                    'max_leaf_nodes', 'min_impurity_split', 'oob_score',
                    'random_state', 'warm_start', 'class_weight',
                    'criterion']

    criterion_dict = {'0': GINI, 'gini': GINI,
                      '1': ENTROPY, 'entropy': ENTROPY,
                      '2': MSE, 'mse': MSE,
                      '3': MAE, 'mae': MAE,
                      '4': POISSON, 'poisson': POISSON,
                      '5': GAMMA, 'gamma': GAMMA,
                      '6': INVERSE_GAUSSIAN,
                      'inverse_gaussian': INVERSE_GAUSSIAN,
                      '7': CRITERION_END}

    classes_ = CumlArrayDescriptor()

    @classmethod
    def _criterion_to_split_criterion(cls, criterion):
        """Translate sklearn-style criterion string to cuML equivalent"""
        if criterion is None:
            split_criterion = cls._default_split_criterion
        else:
            if criterion == "squared_error":
                split_criterion = "mse"
            elif criterion == "absolute_error":
                split_criterion = "mae"
            elif criterion == "poisson":
                split_criterion = "poisson"
            elif criterion == "gini":
                split_criterion = "gini"
            elif criterion == "entropy":
                split_criterion = "entropy"
            else:
                raise NotImplementedError(
                    f'Split criterion {criterion} is not yet supported in'
                    ' cuML. See'
                    ' https://docs.rapids.ai/api/cuml/nightly/api.html#random-forest'
                    ' for full information on supported criteria.'
                )
        return criterion

    @device_interop_preparation
    def __init__(self, *, split_criterion, n_streams=4, n_estimators=100,
                 max_depth=16, handle=None, max_features='sqrt', n_bins=128,
                 bootstrap=True,
                 verbose=False, min_samples_leaf=1, min_samples_split=2,
                 max_samples=1.0, max_leaves=-1, accuracy_metric=None,
                 dtype=None, output_type=None, min_weight_fraction_leaf=None,
                 n_jobs=None, max_leaf_nodes=None, min_impurity_decrease=0.0,
                 min_impurity_split=None, oob_score=None, random_state=None,
                 warm_start=None, class_weight=None,
                 criterion=None,
                 max_batch_size=4096, **kwargs):

        sklearn_params = {"min_weight_fraction_leaf": min_weight_fraction_leaf,
                          "max_leaf_nodes": max_leaf_nodes,
                          "min_impurity_split": min_impurity_split,
                          "oob_score": oob_score, "n_jobs": n_jobs,
                          "warm_start": warm_start,
                          "class_weight": class_weight}

        for key, vals in sklearn_params.items():
            if vals and not cuml.accel.enabled():
                raise TypeError(
                    " The Scikit-learn variable ", key,
                    " is not supported in cuML,"
                    " please read the cuML documentation at "
                    "(https://docs.rapids.ai/api/cuml/nightly/"
                    "api.html#random-forest) for more information")

        for key in kwargs.keys():
            if key not in self._param_names and not cuml.accel.enabled():
                raise TypeError(
                    " The variable ", key,
                    " is not supported in cuML,"
                    " please read the cuML documentation at "
                    "(https://docs.rapids.ai/api/cuml/nightly/"
                    "api.html#random-forest) for more information")

        if handle is None:
            handle = Handle(n_streams=n_streams)

        super(BaseRandomForestModel, self).__init__(
            handle=handle,
            verbose=verbose,
            output_type=output_type)

        if max_depth <= 0:
            raise ValueError("Must specify max_depth >0 ")

        if (str(split_criterion) not in
                BaseRandomForestModel.criterion_dict.keys()):
            warnings.warn("The split criterion chosen was not present"
                          " in the list of options accepted by the model"
                          " and so the CRITERION_END option has been chosen.")
            self.split_criterion = CRITERION_END
        else:
            self.split_criterion = \
                BaseRandomForestModel.criterion_dict[str(split_criterion)]
        if self.split_criterion == MAE:
            raise NotImplementedError(
                "cuML does not currently support mean average error as a"
                " RandomForest split criterion"
            )

        if self.split_criterion == MAE:
            raise NotImplementedError(
                "cuML does not currently support mean average error as a"
                " RandomForest split criterion"
            )

        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.max_samples = max_samples
        self.max_leaves = max_leaves
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_bins = n_bins
        self.n_cols = None
        self.dtype = dtype
        self.accuracy_metric = accuracy_metric
        self.max_batch_size = max_batch_size
        self.n_streams = n_streams
        self.random_state = random_state
        self.rf_forest = 0
        self.rf_forest64 = 0
        self.model_pbuf_bytes = bytearray()
        self.treelite_model = None
        self.treelite_serialized_bytes = None

    def __len__(self):
        """Return the number of estimators in the ensemble."""
        return self.n_estimators

    def _get_max_feat_val(self) -> float:
        if isinstance(self.max_features, int):
            return self.max_features/self.n_cols
        elif isinstance(self.max_features, float):
            return self.max_features
        elif self.max_features == 'sqrt':
            return 1/np.sqrt(self.n_cols)
        elif self.max_features == 'log2':
            return math.log2(self.n_cols)/self.n_cols
        else:
            raise ValueError(
                "Expected `max_features` to be an int, float, or one of ['sqrt', 'log2']."
                "Got {max_features!r} instead."
            )

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

    def cpu_to_gpu(self):
        tl_model = treelite.sklearn.import_model(self._cpu_model)
        self.treelite_serialized_bytes = tl_model.serialize_bytes()
        self.dtype = np.float64
        self.update_labels = False
        super().cpu_to_gpu()
        # Set fitted attributes not transferred by Treelite, but only when the
        # accelerator is active.
        # We only transfer "simple" attributes, not np.ndarrays or DecisionTree
        # instances, as these could be used by the GPU model to make predictions.
        # The list of names below is hand vetted.
        if cuml.accel.enabled():
            for name in ('n_features_in_', 'n_outputs_', 'n_classes_', 'oob_score_'):
                # Not all attributes are always present
                try:
                    value = getattr(self._cpu_model, name)
                except AttributeError:
                    continue
                setattr(self, name, value)

    def gpu_to_cpu(self):
        self._serialize_treelite_bytes()
        tl_model = treelite.Model.deserialize_bytes(self.treelite_serialized_bytes)
        # Make sure the CPU model's hyperparameters are preserved, treelite
        # does not roundtrip hyperparameters.
        params = {}
        if hasattr(self, "_cpu_model"):
            params = self._cpu_model.get_params()

        self._cpu_model = treelite.sklearn.export_model(tl_model)
        self._cpu_model.set_params(**params)

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

        max_feature_val = self._get_max_feat_val()
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
        fil_model = ForestInference(
            treelite_model=treelite_bytes,
            handle=self.handle,
            output_type=self.output_type,
            verbose=self.verbose,
            is_classifier=is_classifier,
            layout=layout,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
        )
        if predict_proba:
            return fil_model.predict_proba(X)
        return fil_model.predict(X, threshold=threshold)

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + BaseRandomForestModel._param_names

    def set_params(self, **params):
        self.treelite_serialized_bytes = None

        super().set_params(**params)
        return self

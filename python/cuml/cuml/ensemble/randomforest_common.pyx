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
import treelite.sklearn

import cuml.accel
from cuml.internals.api_decorators import device_interop_preparation
from cuml.internals.safe_imports import gpu_only_import

cp = gpu_only_import('cupy')
import math
import typing
import warnings

from cuml.internals.safe_imports import cpu_only_import

np = cpu_only_import('numpy')
from pylibraft.common.handle import Handle

import cuml.internals
from cuml.common.exceptions import NotFittedError
from cuml.internals.array import CumlArray
from cuml.internals.base import UniversalBase
from cuml.legacy.fil.fil import ForestInference, TreeliteModel

from cython.operator cimport dereference as deref

from cuml.ensemble.randomforest_shared import (
    treelite_deserialize,
    treelite_serialize,
)

from cuml.ensemble.randomforest_shared cimport *

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

        if ((random_state is not None) and (n_streams != 1)):
            warnings.warn("For reproducible results in Random Forest"
                          " Classifier or for almost reproducible results"
                          " in Random Forest Regressor, n_streams=1 is "
                          "recommended. If n_streams is > 1, results may vary "
                          "due to stream/thread timing differences, even when "
                          "random_state is set")
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
        self.treelite_handle = None
        self.treelite_serialized_model = None

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
        elif self.max_features == 'auto':
            if self.RF_type == CLASSIFICATION:
                warnings.warn(
                    "`max_features='auto'` has been deprecated in 24.06 "
                    "and will be removed in 25.08. To keep the past behaviour "
                    "and silence this warning, explicitly set "
                    "`max_features='sqrt'`.",
                    FutureWarning
                )
                return 1/np.sqrt(self.n_cols)
            else:
                warnings.warn(
                    "`max_features='auto'` has been deprecated in 24.06 "
                    "and will be removed in 25.08. To keep the past behaviour "
                    "and silence this warning, explicitly set "
                    "`max_features=1.0`.",
                    FutureWarning
                )
                return 1.0
        else:
            raise ValueError(
                "Wrong value passed in for max_features"
                " please read the documentation present at "
                "(https://docs.rapids.ai/api/cuml/nightly/api.html"
                "#random-forest)")

    def _get_serialized_model(self):
        """
        Returns the self.model_pbuf_bytes.
        Cuml RF model gets converted to treelite protobuf bytes by:

            * Converting the cuml RF model to a treelite model. The treelite
            models handle (pointer) is returned
            * The treelite model handle is used to convert the treelite model
            to a treelite protobuf model which is stored in a temporary file.
            The protobuf model information is read from the temporary file and
            the byte information is returned.

        The treelite handle is stored `self.treelite_handle` and the treelite
        protobuf model bytes are stored in `self.model_pbuf_bytes`. If either
        of information is already present in the model then the respective
        step is skipped.
        """
        if self.treelite_serialized_model:
            return self.treelite_serialized_model
        elif self.treelite_handle:
            fit_mod_ptr = self.treelite_handle
        else:
            fit_mod_ptr = self._obtain_treelite_handle()
        cdef uintptr_t model_ptr = <uintptr_t> fit_mod_ptr
        self.treelite_serialized_model = treelite_serialize(model_ptr)
        return self.treelite_serialized_model

    def _obtain_treelite_handle(self):
        if (not self.treelite_serialized_model) and (not self.rf_forest):
            raise NotFittedError(
                    "Attempting to create treelite from un-fit forest.")

        cdef TreeliteModelHandle tl_handle = NULL
        if self.treelite_handle:
            return self.treelite_handle  # Use cached version

        elif self.treelite_serialized_model:  # bytes -> Treelite
            tl_handle = <TreeliteModelHandle><uintptr_t>treelite_deserialize(
                self.treelite_serialized_model)

        else:
            if self.dtype not in [np.float32, np.float64]:
                raise ValueError("Unknown dtype.")

            if self.RF_type == CLASSIFICATION:
                if self.dtype==np.float32:
                    build_treelite_forest(
                        &tl_handle,
                        <RandomForestMetaData[float, int]*>
                        <uintptr_t> self.rf_forest,
                        <int> self.n_cols
                        )
                elif self.dtype==np.float64:
                    build_treelite_forest(
                        &tl_handle,
                        <RandomForestMetaData[double, int]*>
                        <uintptr_t> self.rf_forest64,
                        <int> self.n_cols
                        )
            else:
                if self.dtype==np.float32:
                    build_treelite_forest(
                        &tl_handle,
                        <RandomForestMetaData[float, float]*>
                        <uintptr_t> self.rf_forest,
                        <int> self.n_cols
                        )
                elif self.dtype==np.float64:
                    build_treelite_forest(
                        &tl_handle,
                        <RandomForestMetaData[double, double]*>
                        <uintptr_t> self.rf_forest64,
                        <int> self.n_cols
                        )

        self.treelite_handle = <uintptr_t> tl_handle
        return self.treelite_handle

    def cpu_to_gpu(self):
        tl_model = treelite.sklearn.import_model(self._cpu_model)
        self._temp = TreeliteModel.from_treelite_bytes(tl_model.serialize_bytes())
        self.treelite_serialized_model = treelite_serialize(self._temp.handle)
        self._obtain_treelite_handle()
        self.dtype = np.float64
        self.update_labels = False
        super().cpu_to_gpu()
        # Set fitted attributes not transferred by treelite, but only when the
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
        self._obtain_treelite_handle()
        tl_model = TreeliteModel.from_treelite_model_handle(
            self.treelite_handle,
            take_handle_ownership=False)
        tl_bytes = tl_model.to_treelite_bytes()
        tl_model2 = treelite.Model.deserialize_bytes(tl_bytes)
        # Make sure the CPU model's hyper-parameters are preserved, treelite
        # does not roundtrip hyper-parameters.
        params = {}
        if hasattr(self, "_cpu_model"):
            params = self._cpu_model.get_params()

        self._cpu_model = treelite.sklearn.export_model(tl_model2)
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

    def _tl_handle_from_bytes(self, treelite_serialized_model):
        if not treelite_serialized_model:
            raise ValueError(
                '_tl_handle_from_bytes() requires non-empty serialized model')
        return treelite_deserialize(treelite_serialized_model)

    def _concatenate_treelite_handle(self, treelite_handle):
        cdef TreeliteModelHandle concat_model_handle = NULL
        cdef vector[TreeliteModelHandle] *model_handles \
            = new vector[TreeliteModelHandle]()
        cdef uintptr_t mod_ptr
        for i in treelite_handle:
            mod_ptr = <uintptr_t>i
            model_handles.push_back((
                <TreeliteModelHandle> mod_ptr))

        self._reset_forest_data()
        concat_model_handle = concatenate_trees(deref(model_handles))
        cdef uintptr_t concat_model_ptr = <uintptr_t> concat_model_handle
        self.treelite_handle = concat_model_ptr
        self.treelite_serialized_model = treelite_serialize(concat_model_ptr)

        # Fix up some instance variables that should match the new TL model
        tl_model = TreeliteModel.from_treelite_model_handle(
            self.treelite_handle,
            take_handle_ownership=False)
        self.n_cols = tl_model.num_features
        self.n_estimators = tl_model.num_trees

        return self

    def _predict_model_on_gpu(self, X, algo, convert_dtype,
                              fil_sparse_format, threshold=0.5,
                              output_class=False,
                              predict_proba=False) -> CumlArray:
        treelite_handle = self._obtain_treelite_handle()
        storage_type = \
            _check_fil_parameter_validity(depth=self.max_depth,
                                          fil_sparse_format=fil_sparse_format,
                                          algo=algo)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', FutureWarning)
            fil_model = ForestInference(handle=self.handle, verbose=self.verbose,
                                        output_type=self.output_type)
        tl_to_fil_model = \
            fil_model.load_using_treelite_handle(treelite_handle,
                                                 output_class=output_class,
                                                 threshold=threshold,
                                                 algo=algo,
                                                 storage_type=storage_type)

        if (predict_proba):
            preds = tl_to_fil_model.predict_proba(X)
        else:
            preds = tl_to_fil_model.predict(X)
        return preds

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + BaseRandomForestModel._param_names

    def set_params(self, **params):
        self.treelite_serialized_model = None

        super().set_params(**params)
        return self


def _check_fil_parameter_validity(depth, algo, fil_sparse_format):
    """
    Check if the FIL storage format type passed by the user is right
    for the trained cuml Random Forest model they have.

    Parameters
    ----------
    depth : max depth value used to train model
    algo : string (default = 'auto')
        This is optional and required only while performing the
        predict operation on the GPU.

         * ``'naive'`` - simple inference using shared memory
         * ``'tree_reorg'`` - similar to naive but trees rearranged to be more
           coalescing-friendly
         * ``'batch_tree_reorg'`` - similar to tree_reorg but predicting
           multiple rows per thread block
         * ``'auto'`` - choose the algorithm automatically. Currently
         * ``'batch_tree_reorg'`` is used for dense storage
           and 'naive' for sparse storage

    fil_sparse_format : boolean or string (default = 'auto')
        This variable is used to choose the type of forest that will be
        created in the Forest Inference Library. It is not required
        while using predict_model='CPU'.

         * ``'auto'`` - choose the storage type automatically
           (currently True is chosen by auto)
         * ``False`` - create a dense forest
         * ``True`` - create a sparse forest, requires algo='naive'
           or algo='auto'

    Returns
    ----------
    fil_sparse_format
    """
    accepted_fil_spars_format = {True, False, 'auto'}

    if (depth > 16 and (fil_sparse_format is False or
                        algo == 'tree_reorg' or
                        algo == 'batch_tree_reorg')):
        raise ValueError("While creating a forest with max_depth greater "
                         "than 16, `fil_sparse_format` should be True. "
                         "If `fil_sparse_format=False` then the memory"
                         "consumed while creating the FIL forest is very "
                         "large and the process will be aborted. In "
                         "addition, `algo` must be either set to `naive' "
                         "or `auto` to set 'fil_sparse_format=True`.")
    if fil_sparse_format not in accepted_fil_spars_format:
        raise ValueError(
            "The value entered for spares_forest is not "
            "supported. Please refer to the documentation at "
            "(https://docs.rapids.ai/api/cuml/nightly/api.html"
            "#forest-inferencing) to see the accepted values.")
    return fil_sparse_format


def _obtain_fil_model(treelite_handle, depth,
                      output_class=True,
                      threshold=0.5, algo='auto',
                      fil_sparse_format='auto'):
    """
    Creates a Forest Inference (FIL) model using the treelite
    handle obtained from the cuML Random Forest model.

    Returns
    ----------
    fil_model :
        A Forest Inference model which can be used to perform
        inferencing on the random forest model.
    """
    storage_format = \
        _check_fil_parameter_validity(depth=depth,
                                      fil_sparse_format=fil_sparse_format,
                                      algo=algo)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        # Use output_type="input" to prevent an error
        fil_model = ForestInference(output_type="input")

    tl_to_fil_model = \
        fil_model.load_using_treelite_handle(treelite_handle,
                                             output_class=output_class,
                                             threshold=threshold,
                                             algo=algo,
                                             storage_type=storage_format)

    return tl_to_fil_model

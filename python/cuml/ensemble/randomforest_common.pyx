#
# Copyright (c) 2020, NVIDIA CORPORATION.
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

import ctypes
import cupy as cp
import math
import warnings

import numpy as np
from cuml import ForestInference
from cuml.fil.fil import TreeliteModel
from cuml.common.handle import Handle
from cuml.common.base import Base
from cuml.common.array import CumlArray

from cython.operator cimport dereference as deref

from cuml.ensemble.randomforest_shared cimport *
from cuml.common import input_to_cuml_array, rmm_cupy_ary

cimport cython

# create a cdef class and cdef func which will call the C++ cdef func and then return the required handle and stuff
class BaseRandomForestModel(Base):
    variables = ['n_estimators', 'max_depth', 'handle',
                 'max_features', 'n_bins',
                 'split_algo', 'split_criterion', 'min_rows_per_node',
                 'min_impurity_decrease',
                 'bootstrap', 'bootstrap_features',
                 'verbose', 'rows_sample',
                 'max_leaves', 'quantile_per_tree']

    def _create_model(self, model, seed, split_criterion,
                      n_streams, n_estimators=100,
                      max_depth=16, handle=None, max_features='auto',
                      n_bins=8, split_algo=1, bootstrap=True,
                      bootstrap_features=False,
                      verbose=False, min_rows_per_node=2,
                      rows_sample=1.0, max_leaves=-1,
                      accuracy_metric=None, dtype=None,
                      output_type=None, min_samples_leaf=None,
                      min_weight_fraction_leaf=None, n_jobs=None,
                      max_leaf_nodes=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, oob_score=None,
                      random_state=None, warm_start=None, class_weight=None,
                      quantile_per_tree=False, criterion=None):

        if accuracy_metric:
            model.variables.append('accuracy_metric')
        sklearn_params = {"criterion": criterion,
                          "min_samples_leaf": min_samples_leaf,
                          "min_weight_fraction_leaf": min_weight_fraction_leaf,
                          "max_leaf_nodes": max_leaf_nodes,
                          "min_impurity_split": min_impurity_split,
                          "oob_score": oob_score, "n_jobs": n_jobs,
                          "random_state": random_state,
                          "warm_start": warm_start,
                          "class_weight": class_weight}

        for key, vals in sklearn_params.items():
            if vals is not None:
                raise TypeError(" The Scikit-learn variable ", key,
                                " is not supported in cuML,"
                                " please read the cuML documentation for"
                                " more information")

        if handle is None:
            handle = Handle(n_streams)

        super(model, self).__init__(handle=handle,
                                    verbose=verbose,
                                    output_type=output_type)
        if max_depth < 0:
            raise ValueError("Must specify max_depth >0 ")

        self.split_algo = split_algo
        criterion_dict = {'0': GINI, '1': ENTROPY, '2': MSE,
                          '3': MAE, '4': CRITERION_END}
        if str(split_criterion) not in criterion_dict.keys():
            warnings.warn("The split criterion chosen was not present"
                          " in the list of options accepted by the model"
                          " and so the CRITERION_END option has been chosen.")
            self.split_criterion = CRITERION_END
        else:
            self.split_criterion = criterion_dict[str(split_criterion)]

        self.min_rows_per_node = min_rows_per_node
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap_features = bootstrap_features
        self.rows_sample = rows_sample
        self.max_leaves = max_leaves
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.verbose = verbose
        self.n_bins = n_bins
        self.n_cols = None
        self.dtype = dtype
        self.accuracy_metric = accuracy_metric
        self.quantile_per_tree = quantile_per_tree
        self.n_streams = handle.getNumInternalStreams()
        self.seed = seed
        self.rf_forest = 0
        self.rf_forest64 = 0
        self.model_pbuf_bytes = bytearray()
        self.treelite_handle = None

    def _get_max_feat_val(self):
        if type(self.max_features) == int:
            return self.max_features/self.n_cols
        elif type(self.max_features) == float:
            return self.max_features
        elif self.max_features == 'sqrt':
            return 1/np.sqrt(self.n_cols)
        elif self.max_features == 'log2':
            return math.log2(self.n_cols)/self.n_cols
        elif self.max_features == 'auto':
            if self.RF_type == CLASSIFICATION:
                return 1/np.sqrt(self.n_cols)
            else:
                return 1.0
        else:
            raise ValueError("Wrong value passed in for max_features"
                             " please read the documentation")

    def _get_protobuf_bytes(self):
        """
        Returns the self.model_pbuf_bytes.
        Cuml RF model gets converted to treelite protobuf bytes by:
            1. converting the cuml RF model to a treelite model. The treelite
            models handle (pointer) is returned
            2. The treelite model handle is used to convert the treelite model
            to a treelite protobuf model which is stored in a temporary file.
            The protobuf model information is read from the temporary file and
            the byte information is returned.
        The treelite handle is stored `self.treelite_handle` and the treelite
        protobuf model bytes are stored in `self.model_pbuf_bytes`. If either
        of information is already present in the model then the respective
        step is skipped.
        """
        if self.model_pbuf_bytes:
            return self.model_pbuf_bytes
        elif self.treelite_handle:
            fit_mod_ptr = self.treelite_handle
        else:
            fit_mod_ptr = self._obtain_treelite_handle()
        cdef uintptr_t model_ptr = <uintptr_t> fit_mod_ptr
        cdef vector[unsigned char] pbuf_mod_info = \
            save_model(<ModelHandle> model_ptr)
        cdef unsigned char[::1] pbuf_mod_view = \
            <unsigned char[:pbuf_mod_info.size():1]>pbuf_mod_info.data()
        self.model_pbuf_bytes = bytearray(memoryview(pbuf_mod_view))
        return self.model_pbuf_bytes


    def _obtain_treelite_handle(self):
        if self.treelite_handle:
            print(" treelite handle in obt : ", self.treelite_handle)
            return self.treelite_handle # Use cached version
        cdef ModelHandle cuml_model_ptr = NULL
        cdef unsigned char[::1] model_pbuf_mv
        cdef vector[unsigned char] model_pbuf_vec
        if self.model_pbuf_bytes:
            model_pbuf_mv = self.model_pbuf_bytes
            with cython.boundscheck(False):
                model_pbuf_vec.assign(& model_pbuf_mv[0],
                                      & model_pbuf_mv[model_pbuf_mv.shape[0]])
        else:
            model_pbuf_vec = <vector[unsigned char]&> bytearray()
        if self.RF_type == CLASSIFICATION:
            build_treelite_forest(
                & cuml_model_ptr,
                <RandomForestMetaData[float, int]*><size_t> self.rf_forest,
                <int> self.n_cols,
                <int> self.num_classes,
                model_pbuf_vec)
        else:
            build_treelite_forest(
                & cuml_model_ptr,
                <RandomForestMetaData[float, float]*><size_t> self.rf_forest,
                <int> self.n_cols,
                <int> REGRESSION_MODEL,
                model_pbuf_vec)

        mod_ptr = <uintptr_t> cuml_model_ptr
        self.treelite_handle = ctypes.c_void_p(mod_ptr).value
        return self.treelite_handle

    def _dataset_setup(self, X, y, convert_dtype):
        self._set_output_type(X)

        # Reset the old tree data for new fit call
        self._reset_forest_data()

        X_m, self.n_rows, self.n_cols, self.dtype = \
            input_to_cuml_array(X, check_dtype=[np.float32, np.float64],
                                order='F')
        if self.n_bins > self.n_rows:
            raise ValueError("The number of bins,`n_bins` can not be greater"
                             " than the number of samples used for training.")
        if self.RF_type == CLASSIFICATION:
            y_m, _, _, y_dtype = \
                input_to_cuml_array(y, check_dtype=np.int32,
                                    convert_to_dtype=(np.int32 if convert_dtype
                                                      else None),
                                    check_rows=self.n_rows, check_cols=1)
            if y_dtype != np.int32:
                raise TypeError("The labels `y` need to be of dtype `np.int32`")
            unique_labels = rmm_cupy_ary(cp.unique, y_m)
            self.num_classes = len(unique_labels)
            for i in range(self.num_classes):
                if i not in unique_labels:
                    raise ValueError("The labels need "
                                     "to be consecutive values from "
                                     "0 to the number of unique label values")
        else:
            y_m, _, _, y_dtype = \
                input_to_cuml_array(y,
                                    convert_to_dtype=(self.dtype if convert_dtype
                                                      else None),
                                    check_rows=self.n_rows, check_cols=1)

        if self.dtype == np.float64:
            warnings.warn("To use GPU-based prediction, first train using \
                          float 32 data to fit the estimator.")

        max_feature_val = self._get_max_feat_val()
        if type(self.min_rows_per_node) == float:
            self.min_rows_per_node = math.ceil(self.min_rows_per_node*self.n_rows)
        del X
        del y
        return X_m, y_m, max_feature_val
    
    def _tl_model_handles(self, model_bytes):
        cdef ModelHandle cuml_model_ptr = NULL
        if self.RF_type == CLASSIFICATION:
            build_treelite_forest(
                & cuml_model_ptr,
                <RandomForestMetaData[float, int]*><size_t> self.rf_forest,
                <int> self.n_cols,
                <int> self.num_classes,
                <vector[unsigned char] &> model_bytes)
        else:
            build_treelite_forest(
                & cuml_model_ptr,
                <RandomForestMetaData[float, float]*><size_t> self.rf_forest,
                <int> self.n_cols,
                <int> REGRESSION_MODEL,
                <vector[unsigned char] &> model_bytes)
        mod_handle = <uintptr_t> cuml_model_ptr

        return ctypes.c_void_p(mod_handle).value


    def _concatenate_treelite_handle(self, treelite_handle):
        cdef ModelHandle concat_model_handle = NULL
        cdef vector[ModelHandle] *model_handles \
            = new vector[ModelHandle]()
        cdef uintptr_t mod_ptr
        for i in treelite_handle:
            mod_ptr = <uintptr_t>i
            model_handles.push_back((
                <ModelHandle> mod_ptr))
        self._reset_forest_data()
        concat_model_handle = concatenate_trees(deref(model_handles))
        cdef uintptr_t concat_model_ptr = <uintptr_t> concat_model_handle
        self.treelite_handle = concat_model_ptr
        cdef vector[unsigned char] pbuf_mod_info = \
            save_model(<ModelHandle> concat_model_ptr)
        cdef unsigned char[::1] pbuf_mod_view = \
            <unsigned char[:pbuf_mod_info.size():1]>pbuf_mod_info.data()
        self.model_pbuf_bytes = bytearray(memoryview(pbuf_mod_view))

        # Fix up some instance variables that should match the new TL model
        tl_model = TreeliteModel.from_treelite_model_handle(
            self.treelite_handle,
            take_handle_ownership=False)
        self.n_cols = tl_model.num_features
        self.n_estimators = tl_model.num_trees
        return self

    
    def _predict_model_on_gpu(self, X, algo, convert_dtype,
                              fil_sparse_format, threshold=0.5,
                              output_class=False, predict_proba=False):
        out_type = self._get_output_type(X)
        cdef ModelHandle cuml_model_ptr = NULL
        _, n_rows, n_cols, dtype = \
            input_to_cuml_array(X, order='F',
                                check_cols=self.n_cols)

        if dtype == np.float64 and not convert_dtype:
            raise TypeError("GPU based predict only accepts np.float32 data. \
                            Please set convert_dtype=True to convert the test \
                            data to the same dtype as the data used to train, \
                            ie. np.float32. If you would like to use test \
                            data of dtype=np.float64 please set \
                            predict_model='CPU' to use the CPU implementation \
                            of predict.")

        treelite_handle = self._obtain_treelite_handle()

        storage_type = \
            _check_fil_parameter_validity(depth=self.max_depth,
                                          fil_sparse_format=fil_sparse_format,
                                          algo=algo)
        fil_model = ForestInference()
        tl_to_fil_model = \
            fil_model.load_using_treelite_handle(treelite_handle,
                                                 output_class=output_class,
                                                 threshold=threshold,
                                                 algo=algo,
                                                 storage_type=storage_type)

        preds = tl_to_fil_model.predict(X, output_type=out_type,
                                        predict_proba=predict_proba)
        return preds
    
    def _get_params(self, model, deep):
        params = dict()
        for key in model.variables:
            if key in ['handle']:
                continue
            var_value = getattr(self, key, None)
            params[key] = var_value
        return params

    def _set_params(self, model, **params):
        self.handle.__setstate__(self.n_streams)
        self.model_pbuf_bytes = []

        if not params:
            return self
        for key, value in params.items():
            if key not in model.variables:
                raise ValueError('Invalid parameter for estimator')
            else:
                setattr(self, key, value)
        return self

def _check_fil_parameter_validity(depth, algo, fil_sparse_format):
    storage_format = _check_fil_sparse_format_value(fil_sparse_format)
    if (depth > 16 and (storage_format == 'dense' or
                        algo == 'tree_reorg' or
                        algo == 'batch_tree_reorg')):
        raise ValueError("While creating a forest with max_depth greater "
                         "than 16, `fil_sparse_format` should be True. "
                         "If `fil_sparse_format=False` then the memory"
                         "consumed while creating the FIL forest is very "
                         "large and the process will be aborted. In "
                         "addition, `algo` must be either set to `naive' "
                         "or `auto` to set 'fil_sparse_format=True`.")
    return storage_format


def _check_fil_sparse_format_value(fil_sparse_format):
    accepted_vals = [True, False, 'auto']
    if fil_sparse_format == 'auto':
        storage_format = fil_sparse_format
    elif not fil_sparse_format:
        storage_format = 'dense'
    elif fil_sparse_format not in accepted_vals:
        raise ValueError("The value entered for spares_forest is not "
                         "supported. Please refer to the documentation "
                         "to see the accepted values.")
    else:
        storage_format = 'sparse'
    return storage_format


def _obtain_treelite_model(treelite_handle):
    """
    Creates a Treelite model using the treelite handle
    obtained from the cuML Random Forest model.

    Returns
    ----------
    tl_to_fil_model : Treelite version of this model
    """
    treelite_model = \
        TreeliteModel.from_treelite_model_handle(treelite_handle)
    return treelite_model


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

    fil_model = ForestInference()
    tl_to_fil_model = \
        fil_model.load_using_treelite_handle(treelite_handle,
                                             output_class=output_class,
                                             threshold=threshold,
                                             algo=algo,
                                             storage_type=storage_format)

    return tl_to_fil_model

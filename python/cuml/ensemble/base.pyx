import ctypes
import cupy as cp
import math
import warnings

import numpy as np
from cuml import ForestInference
from cuml.fil.fil import TreeliteModel as tl
from cuml.common.handle import Handle
from cuml.common.base import Base
from cuml.common.array import CumlArray

from cython.operator cimport dereference as deref

from cuml.ensemble.randomforest_shared cimport *
from cuml.common import input_to_cuml_array, rmm_cupy_ary



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
        self.model_pbuf_bytes = bytearray()
        self.treelite_handle = None

    def _dataset_setup(self, X, y, convert_dtype):
        self._set_output_type(X)

        # Reset the old tree data for new fit call
        self._reset_forest_data()

        #cdef uintptr_t X_ptr, y_ptr

        X_m, self.n_rows, self.n_cols, self.dtype = \
            input_to_cuml_array(X, check_dtype=[np.float32, np.float64],
                                order='F')
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

        return X_m, y_m, max_feature_val

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
    print(" storage_format : ", storage_format)
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
    print(" storage_format : ", storage_format)
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
        tl.from_treelite_model_handle(treelite_handle)
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
    print(" treelite handle in obt fil : ", treelite_handle)
    storage_format = \
        _check_fil_parameter_validity(depth=depth,
                                      fil_sparse_format=fil_sparse_format,
                                      algo=algo)

    fil_model = ForestInference()
    tl_to_fil_model = \
        fil_model.load_from_randomforest(treelite_handle,
                                         output_class=output_class,
                                         threshold=threshold,
                                         algo=algo,
                                         storage_type=storage_format)

    return tl_to_fil_model

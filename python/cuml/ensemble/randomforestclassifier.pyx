#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.
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


# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import ctypes
import cudf
import cupy as cp
import math
import numpy as np
import rmm
import warnings

from libcpp cimport bool
from libcpp.vector cimport vector
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from cuml import ForestInference
from cuml.common.base import Base
from cuml.common.handle import Handle
from cuml.common.handle cimport cumlHandle
from cuml.ensemble.randomforest_shared cimport *
from cuml.utils import get_cudf_column_ptr, get_dev_array_ptr, \
    input_to_dev_array, zeros, rmm_cupy_ary

from numba import cuda

cimport cuml.common.handle
cimport cuml.common.cuda

cdef extern from "cuml/ensemble/randomforest.hpp" namespace "ML":

    cdef void fit(cumlHandle & handle,
                  RandomForestMetaData[float, int]*,
                  float*,
                  int,
                  int,
                  int*,
                  int,
                  RF_params) except +

    cdef void fit(cumlHandle & handle,
                  RandomForestMetaData[double, int]*,
                  double*,
                  int,
                  int,
                  int*,
                  int,
                  RF_params) except +

    cdef void predict(cumlHandle& handle,
                      RandomForestMetaData[float, int] *,
                      float*,
                      int,
                      int,
                      int*,
                      bool) except +

    cdef void predict(cumlHandle& handle,
                      RandomForestMetaData[double, int]*,
                      double*,
                      int,
                      int,
                      int*,
                      bool) except +

    cdef void predictGetAll(cumlHandle& handle,
                            RandomForestMetaData[float, int] *,
                            float*,
                            int,
                            int,
                            int*,
                            bool) except +

    cdef void predictGetAll(cumlHandle& handle,
                            RandomForestMetaData[double, int]*,
                            double*,
                            int,
                            int,
                            int*,
                            bool) except +

    cdef RF_metrics score(cumlHandle& handle,
                          RandomForestMetaData[float, int]*,
                          int*,
                          int,
                          int*,
                          bool) except +

    cdef RF_metrics score(cumlHandle& handle,
                          RandomForestMetaData[double, int]*,
                          int*,
                          int,
                          int*,
                          bool) except +


class RandomForestClassifier(Base):
    """
    Implements a Random Forest classifier model which fits multiple decision
    tree classifiers in an ensemble.

    Note that the underlying algorithm for tree node splits differs from that
    used in scikit-learn. By default, the cuML Random Forest uses a
    histogram-based algorithms to determine splits, rather than an exact
    count. You can tune the size of the histograms with the n_bins parameter.

    **Known Limitations**: This is an early release of the cuML
    Random Forest code. It contains a few known limitations:

       * GPU-based inference is only supported if the model was trained
         with 32-bit (float32) datatypes. CPU-based inference may be used
         in this case as a slower fallback.
       * Very deep / very wide models may exhaust available GPU memory.
         Future versions of cuML will provide an alternative algorithm to
         reduce memory consumption.

    Examples
    ---------
    .. code-block:: python

            import numpy as np
            from cuml.ensemble import RandomForestClassifier as cuRFC

            X = np.random.normal(size=(10,4)).astype(np.float32)
            y = np.asarray([0,1]*5, dtype=np.int32)

            cuml_model = cuRFC(max_features=1.0,
                               n_bins=8,
                               n_estimators=40)
            cuml_model.fit(X,y)
            cuml_predict = cuml_model.predict(X)

            print("Predicted labels : ", cuml_predict)

    Output:

    .. code-block:: none

            Predicted labels :  [0 1 0 1 0 1 0 1 0 1]

    Parameters
    -----------
    n_estimators : int (default = 100)
        Number of trees in the forest. (Default changed to 100 in cuML 0.11)
    handle : cuml.Handle
        If it is None, a new one is created just for this class.
    split_criterion: The criterion used to split nodes.
        0 for GINI, 1 for ENTROPY
        2 and 3 not valid for classification
        (default = 0)
    split_algo : int (default = 1)
        The algorithm to determine how nodes are split in the tree.
        0 for HIST and 1 for GLOBAL_QUANTILE. HIST curently uses a slower
        tree-building algorithm so GLOBAL_QUANTILE is recommended for most
        cases.
    bootstrap : boolean (default = True)
        Control bootstrapping.
        If True, each tree in the forest is built
        on a bootstrapped sample with replacement.
        If False, sampling without replacement is done.
    bootstrap_features : boolean (default = False)
        Control bootstrapping for features.
        If features are drawn with or without replacement
    rows_sample : float (default = 1.0)
        Ratio of dataset rows used while fitting each tree.
    max_depth : int (default = 16)
        Maximum tree depth. Unlimited (i.e, until leaves are pure),
        if -1. Unlimited depth is not supported with split_algo=1.
        *Note that this default differs from scikit-learn's
        random forest, which defaults to unlimited depth.*
    max_leaves : int (default = -1)
        Maximum leaf nodes per tree. Soft constraint. Unlimited,
        if -1.
    max_features : int, float, or string (default = 'auto')
        Ratio of number of features (columns) to consider per node split.
        If int then max_features/n_features.
        If float then max_features is used as a fraction.
        If 'auto' then max_features=1/sqrt(n_features).
        If 'sqrt' then max_features=1/sqrt(n_features).
        If 'log2' then max_features=log2(n_features)/n_features.
    n_bins :  int (default = 8)
        Number of bins used by the split algorithm.
    min_rows_per_node : int or float (default = 2)
        The minimum number of samples (rows) needed to split a node.
        If int then number of sample rows
        If float the min_rows_per_sample*n_rows
    min_impurity_decrease : float (default = 0.0)
        Minimum decrease in impurity requried for
        node to be spilt.
    quantile_per_tree : boolean (default = False)
        Whether quantile is computed for individal trees in RF.
        Only relevant for GLOBAL_QUANTILE split_algo.
    seed : int (default = None)
        Seed for the random number generator. Unseeded by default.
    """

    variables = ['n_estimators', 'max_depth', 'handle',
                 'max_features', 'n_bins',
                 'split_algo', 'split_criterion', 'min_rows_per_node',
                 'min_impurity_decrease',
                 'bootstrap', 'bootstrap_features',
                 'verbose', 'rows_sample',
                 'max_leaves', 'quantile_per_tree']

    def __init__(self, n_estimators=100, max_depth=16, handle=None,
                 max_features='auto', n_bins=8, n_streams=8,
                 split_algo=1, split_criterion=0, min_rows_per_node=2,
                 bootstrap=True, bootstrap_features=False,
                 type_model="classifier", verbose=False,
                 rows_sample=1.0, max_leaves=-1, quantile_per_tree=False,
                 gdf_datatype=None, criterion=None,
                 min_samples_leaf=None, min_weight_fraction_leaf=None,
                 max_leaf_nodes=None, min_impurity_decrease=0.0,
                 min_impurity_split=None, oob_score=None, n_jobs=None,
                 random_state=None, warm_start=None, class_weight=None,
                 seed=None):

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
                raise TypeError("The Scikit-learn variable", key,
                                " is not supported in cuML,"
                                " please read the cuML documentation for"
                                " more information")

        if max_depth < 0:
            raise ValueError("Must specify max_depth >0")

        if handle is None:
            handle = Handle(n_streams)

        super(RandomForestClassifier, self).__init__(handle, verbose)

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
        self.quantile_per_tree = quantile_per_tree
        self.n_cols = None
        self.dtype = None
        self.n_streams = handle.getNumInternalStreams()
        self.seed = seed
        if ((seed is not None) and (n_streams != 1)):
            warnings.warn("Random seed requires n_streams=1.")
        self.model_pbuf_bytes = []
        cdef RandomForestMetaData[float, int] *rf_forest = \
            new RandomForestMetaData[float, int]()
        self.rf_forest = <size_t> rf_forest
        cdef RandomForestMetaData[double, int] *rf_forest64 = \
            new RandomForestMetaData[double, int]()
        self.rf_forest64 = <size_t> rf_forest64
    """
    TODO:
        Add the preprocess and postprocess functions
        in the cython code to normalize the labels
        Link to the above issue on github :
        https://github.com/rapidsai/cuml/issues/691
    """
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['handle']
        if self.n_cols:
            # only if model has been fit previously
            self.model_pbuf_bytes = self._get_model_info()
        cdef size_t params_t = <size_t> self.rf_forest
        cdef  RandomForestMetaData[float, int] *rf_forest = \
            <RandomForestMetaData[float, int]*>params_t

        cdef size_t params_t64 = <size_t> self.rf_forest64
        cdef  RandomForestMetaData[double, int] *rf_forest64 = \
            <RandomForestMetaData[double, int]*>params_t64

        state["verbose"] = self.verbose
        state["model_pbuf_bytes"] = self.model_pbuf_bytes

        if self.dtype == np.float32:
            state["rf_params"] = rf_forest.rf_params
        else:
            state["rf_params64"] = rf_forest64.rf_params
        return state

    def __setstate__(self, state):

        super(RandomForestClassifier, self).__init__(handle=None,
                                                     verbose=state['verbose'])
        cdef  RandomForestMetaData[float, int] *rf_forest = \
            new RandomForestMetaData[float, int]()
        cdef  RandomForestMetaData[double, int] *rf_forest64 = \
            new RandomForestMetaData[double, int]()

        self.model_pbuf_bytes = state["model_pbuf_bytes"]

        if state["dtype"] == np.float32:
            rf_forest.rf_params = state["rf_params"]
            state["rf_forest"] = <size_t>rf_forest
        else:
            rf_forest64.rf_params = state["rf_params64"]
            state["rf_forest64"] = <size_t>rf_forest64

        self.__dict__.update(state)

    def __del__(self):
        if self.dtype == np.float32:
            free(<RandomForestMetaData[float, int]*><size_t> self.rf_forest)
        else:
            free(<RandomForestMetaData[double, int]*><size_t> self.rf_forest64)

    def _get_max_feat_val(self):
        if type(self.max_features) == int:
            return self.max_features/self.n_cols
        elif type(self.max_features) == float:
            return self.max_features
        elif self.max_features == 'sqrt' or self.max_features == 'auto':
            return 1/np.sqrt(self.n_cols)
        elif self.max_features == 'log2':
            return math.log2(self.n_cols)/self.n_cols
        else:
            raise ValueError("Wrong value passed in for max_features"
                             " please read the documentation")

    def _get_model_info(self):
        cdef ModelHandle cuml_model_ptr = NULL
        task_category = 1
        cdef RandomForestMetaData[float, int] *rf_forest = \
            <RandomForestMetaData[float, int]*><size_t> self.rf_forest
        build_treelite_forest(& cuml_model_ptr,
                              rf_forest,
                              <int> self.n_cols,
                              <int> task_category,
                              <vector[unsigned char] &> self.model_pbuf_bytes)

        mod_ptr = <size_t> cuml_model_ptr
        fit_mod_ptr = ctypes.c_void_p(mod_ptr).value
        cdef uintptr_t model_ptr = <uintptr_t> fit_mod_ptr
        model_protobuf_bytes = save_model(<ModelHandle> model_ptr)

        return model_protobuf_bytes

    def fit(self, X, y):
        """
        Perform Random Forest Classification on the input data

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy
        y : array-like (device or host) shape = (n_samples, 1)
            Dense vector (int32) of shape (n_samples, 1).
            Acceptable formats: NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy
            These labels should be contiguous integers from 0 to n_classes.
        """
        cdef uintptr_t X_ptr, y_ptr
        y_m, y_ptr, _, _, y_dtype = input_to_dev_array(y)

        if y_dtype != np.int32:
            raise TypeError("The labels `y` need to be of dtype `np.int32`")

        X_m, X_ptr, n_rows, self.n_cols, self.dtype = \
            input_to_dev_array(X, order='F')

        if self.dtype == np.float64:
            warnings.warn("To use GPU-based prediction, first train \
                          using float 32 data to fit the estimator.")

        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()

        unique_labels = rmm_cupy_ary(cp.unique, y_m)
        num_unique_labels = len(unique_labels)

        for i in range(num_unique_labels):
            if i not in unique_labels:
                raise ValueError("The labels need "
                                 "to be consecutive values from "
                                 "0 to the number of unique label values")

        max_feature_val = self._get_max_feat_val()
        if type(self.min_rows_per_node) == float:
            self.min_rows_per_node = math.ceil(self.min_rows_per_node*n_rows)

        cdef RandomForestMetaData[float, int] *rf_forest = \
            <RandomForestMetaData[float, int]*><size_t> self.rf_forest
        cdef RandomForestMetaData[double, int] *rf_forest64 = \
            <RandomForestMetaData[double, int]*><size_t> self.rf_forest64
        if self.seed is None:
            seed_val = <uintptr_t>NULL
        else:
            seed_val = <uintptr_t>self.seed

        rf_params = set_rf_class_obj(<int> self.max_depth,
                                     <int> self.max_leaves,
                                     <float> max_feature_val,
                                     <int> self.n_bins,
                                     <int> self.split_algo,
                                     <int> self.min_rows_per_node,
                                     <float> self.min_impurity_decrease,
                                     <bool> self.bootstrap_features,
                                     <bool> self.bootstrap,
                                     <int> self.n_estimators,
                                     <float> self.rows_sample,
                                     <int> seed_val,
                                     <CRITERION> self.split_criterion,
                                     <bool> self.quantile_per_tree,
                                     <int> self.n_streams)

        if self.dtype == np.float32:
            fit(handle_[0],
                rf_forest,
                <float*> X_ptr,
                <int> n_rows,
                <int> self.n_cols,
                <int*> y_ptr,
                <int> num_unique_labels,
                rf_params)

        elif self.dtype == np.float64:
            rf_params64 = rf_params
            fit(handle_[0],
                rf_forest64,
                <double*> X_ptr,
                <int> n_rows,
                <int> self.n_cols,
                <int*> y_ptr,
                <int> num_unique_labels,
                rf_params64)

        else:
            raise TypeError("supports only np.float32 and np.float64 input,"
                            " but input of type '%s' passed."
                            % (str(self.dtype)))
        # make sure that the `fit` is complete before the following delete
        # call happens
        self.handle.sync()
        del(X_m)
        del(y_m)
        return self

    def _predict_model_on_gpu(self, X, output_class,
                              threshold, algo,
                              num_classes, convert_dtype):
        cdef ModelHandle cuml_model_ptr = NULL
        X_m, _, n_rows, n_cols, X_type = \
            input_to_dev_array(X, order='C', check_dtype=self.dtype,
                               convert_to_dtype=(self.dtype if convert_dtype
                                                 else None),
                               check_cols=self.n_cols)

        cdef RandomForestMetaData[float, int] *rf_forest = \
            <RandomForestMetaData[float, int]*><size_t> self.rf_forest

        build_treelite_forest(& cuml_model_ptr,
                              rf_forest,
                              <int> n_cols,
                              <int> num_classes,
                              <vector[unsigned char] &> self.model_pbuf_bytes)
        mod_ptr = <size_t> cuml_model_ptr
        treelite_handle = ctypes.c_void_p(mod_ptr).value
        fil_model = ForestInference()
        tl_to_fil_model = \
            fil_model.load_from_randomforest(treelite_handle,
                                             output_class=output_class,
                                             threshold=threshold,
                                             algo=algo)
        preds = tl_to_fil_model.predict(X_m)
        del(X_m)
        return preds

    def _predict_model_on_cpu(self, X, convert_dtype):
        cdef uintptr_t X_ptr
        X_m, X_ptr, n_rows, n_cols, _ = \
            input_to_dev_array(X, order='C', check_dtype=self.dtype,
                               convert_to_dtype=(self.dtype if convert_dtype
                                                 else None),
                               check_cols=self.n_cols)

        preds = cudf.Series(zeros(n_rows, dtype=np.int32))
        cdef uintptr_t preds_ptr = get_cudf_column_ptr(preds)

        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()

        cdef RandomForestMetaData[float, int] *rf_forest = \
            <RandomForestMetaData[float, int]*><size_t> self.rf_forest

        cdef RandomForestMetaData[double, int] *rf_forest64 = \
            <RandomForestMetaData[double, int]*><size_t> self.rf_forest64
        if self.dtype == np.float32:
            predict(handle_[0],
                    rf_forest,
                    <float*> X_ptr,
                    <int> n_rows,
                    <int> n_cols,
                    <int*> preds_ptr,
                    <bool> self.verbose)

        elif self.dtype == np.float64:
            predict(handle_[0],
                    rf_forest64,
                    <double*> X_ptr,
                    <int> n_rows,
                    <int> n_cols,
                    <int*> preds_ptr,
                    <bool> self.verbose)
        else:
            raise TypeError("supports only np.float32 and np.float64 input,"
                            " but input of type '%s' passed."
                            % (str(self.dtype)))

        self.handle.sync()
        # synchronous w/o a stream
        predicted_result = preds.to_array()
        del(X_m)
        return predicted_result

    def predict(self, X, predict_model="GPU",
                output_class=True, threshold=0.5,
                algo='BATCH_TREE_REORG',
                num_classes=2, convert_dtype=True):
        """
        Predicts the labels for X.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy
        predict_model : String (default = 'GPU')
            'GPU' to predict using the GPU, 'CPU' otherwise. The GPU can only
            be used if the model was trained on float32 data and `X` is float32
            or convert_dtype is set to True.
        output_class: boolean (default = True)
            This is optional and required only while performing the
            predict operation on the GPU.
            If true, return a 1 or 0 depending on whether the raw
            prediction exceeds the threshold. If False, just return
            the raw prediction.
        algo : string (default = 'BATCH_TREE_REORG')
            This is optional and required only while performing the
            predict operation on the GPU.
            'NAIVE' - simple inference using shared memory
            'TREE_REORG' - similar to naive but trees rearranged to be more
                           coalescing-friendly
            'BATCH_TREE_REORG' - similar to TREE_REORG but predicting
                                 multiple rows per thread block
        threshold : float (default = 0.5)
            Threshold used for classification. Optional and required only
            while performing the predict operation on the GPU.
            It is applied if output_class == True, else it is ignored
        num_classes : int (default = 2)
                      number of different classes present in the dataset
        convert_dtype : bool, optional (default = True)
            When set to True, the predict method will, when necessary, convert
            the input to the data type which was used to train the model. This
            will increase memory used for the method.

        Returns
        ----------
        y : NumPy
           Dense vector (int) of shape (n_samples, 1)
        """

        if predict_model == "CPU":
            preds = self._predict_model_on_cpu(X, convert_dtype)

        elif self.dtype == np.float64 and not convert_dtype:
            raise TypeError("GPU based predict only accepts np.float32 data. \
                            In order use the GPU predict the model should \
                            also be trained using a np.float32 dataset. \
                            If you would like to use np.float64 dtype \
                            then please use the CPU based predict by \
                            setting predict_model = 'CPU'")

        else:
            preds = self._predict_model_on_gpu(X, output_class,
                                               threshold, algo,
                                               num_classes, convert_dtype)

        return preds

    def _predict_get_all(self, X, convert_dtype=True):
        """
        Predicts the labels for X.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        Returns
        ----------
        y : NumPy
           Dense vector (int) of shape (n_samples, 1)
        """
        cdef uintptr_t X_ptr, preds_ptr
        X_m, X_ptr, n_rows, n_cols, _ = \
            input_to_dev_array(X, order='C', check_dtype=self.dtype,
                               convert_to_dtype=(self.dtype if convert_dtype
                                                 else None),
                               check_cols=self.n_cols)

        preds = cudf.Series(zeros(n_rows * self.n_estimators, dtype=np.int32))
        preds_ptr = get_cudf_column_ptr(preds)

        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()

        cdef RandomForestMetaData[float, int] *rf_forest = \
            <RandomForestMetaData[float, int]*><size_t> self.rf_forest

        cdef RandomForestMetaData[double, int] *rf_forest64 = \
            <RandomForestMetaData[double, int]*><size_t> self.rf_forest64

        if self.dtype == np.float32:
            predictGetAll(handle_[0],
                          rf_forest,
                          <float*> X_ptr,
                          <int> n_rows,
                          <int> n_cols,
                          <int*> preds_ptr,
                          <bool> self.verbose)

        elif self.dtype == np.float64:
            predictGetAll(handle_[0],
                          rf_forest64,
                          <double*> X_ptr,
                          <int> n_rows,
                          <int> n_cols,
                          <int*> preds_ptr,
                          <bool> self.verbose)
        else:
            raise TypeError("supports only np.float32 and np.float64 input,"
                            " but input of type '%s' passed."
                            % (str(self.dtype)))

        self.handle.sync()
        predicted_result = preds.to_array()
        del(X_m)
        return predicted_result

    def score(self, X, y, threshold=0.5,
              algo='BATCH_TREE_REORG', num_classes=2,
              convert_dtype=True):
        """
        Calculates the accuracy metric score of the model for X.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy
        y : NumPy
           Dense vector (int) of shape (n_samples, 1)
        algo : string name of the algo from (from algo_t enum)
            This is optional and required only while performing the
            predict operation on the GPU.
            'NAIVE' - simple inference using shared memory
            'TREE_REORG' - similar to naive but trees rearranged to be more
            coalescing-friendly
            'BATCH_TREE_REORG' - similar to TREE_REORG but predicting
            multiple rows per thread block
        threshold : float
            threshold is used to for classification
            This is optional and required only while performing the
            predict operation on the GPU.
        num_classes : integer
            number of different classes present in the dataset
        convert_dtype : boolean, default=True
            whether to convert input data to correct dtype automatically

        Returns
        -------
        float
           Accuracy of the model [0.0 - 1.0]
        """
        cdef uintptr_t X_ptr, y_ptr
        y_m, y_ptr, n_rows, _, y_dtype = \
            input_to_dev_array(y, check_dtype=np.int32,
                               convert_to_dtype=(np.int32 if convert_dtype
                                                 else False))

        preds = self.predict(X, output_class=True,
                             threshold=threshold, algo=algo,
                             num_classes=num_classes,
                             convert_dtype=convert_dtype)

        cdef uintptr_t preds_ptr
        preds_m, preds_ptr, _, _, _ = \
            input_to_dev_array(preds, convert_to_dtype=np.int32)

        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()

        cdef RandomForestMetaData[float, int] *rf_forest = \
            <RandomForestMetaData[float, int]*><size_t> self.rf_forest

        cdef RandomForestMetaData[double, int] *rf_forest64 = \
            <RandomForestMetaData[double, int]*><size_t> self.rf_forest64

        if self.dtype == np.float32:
            self.stats = score(handle_[0],
                               rf_forest,
                               <int*> y_ptr,
                               <int> n_rows,
                               <int*> preds_ptr,
                               <bool> self.verbose)
        elif self.dtype == np.float64:
            self.stats = score(handle_[0],
                               rf_forest64,
                               <int*> y_ptr,
                               <int> n_rows,
                               <int*> preds_ptr,
                               <bool> self.verbose)
        else:
            raise TypeError("supports only np.float32 and np.float64 input,"
                            " but input of type '%s' passed."
                            % (str(self.dtype)))

        self.handle.sync()
        del(y_m)
        del(preds_m)
        return self.stats['accuracy']

    def get_params(self, deep=True):
        """
        Returns the value of all parameters
        required to configure this estimator as a dictionary.

        Parameters
        -----------
        deep : boolean (default = True)
        """

        params = dict()
        for key in RandomForestClassifier.variables:
            if key in ['handle']:
                continue
            var_value = getattr(self, key, None)
            params[key] = var_value
        return params

    def set_params(self, **params):
        """
        Sets the value of parameters required to
        configure this estimator, it functions similar to
        the sklearn set_params.

        Parameters
        -----------
        params : dict of new params
        """
        if not params:
            return self
        for key, value in params.items():
            if key not in RandomForestClassifier.variables:
                raise ValueError('Invalid parameter for estimator')
            else:
                setattr(self, key, value)
        self.__init__()
        return self

    def print_summary(self):
        """
        prints the summary of the forest used to train and test the model
        """
        cdef RandomForestMetaData[float, int] *rf_forest = \
            <RandomForestMetaData[float, int]*><size_t> self.rf_forest

        cdef RandomForestMetaData[double, int] *rf_forest64 = \
            <RandomForestMetaData[double, int]*><size_t> self.rf_forest64

        if self.dtype == np.float64:
            print_rf_summary(rf_forest64)
        else:
            print_rf_summary(rf_forest)

    def print_detailed(self):
        """
        prints the detailed information about the forest used to
        train and test the Random Forest model
        """
        cdef RandomForestMetaData[float, int] *rf_forest = \
            <RandomForestMetaData[float, int]*><size_t> self.rf_forest

        cdef RandomForestMetaData[double, int] *rf_forest64 = \
            <RandomForestMetaData[double, int]*><size_t> self.rf_forest64

        if self.dtype == np.float64:
            print_rf_detailed(rf_forest64)
        else:
            print_rf_detailed(rf_forest)

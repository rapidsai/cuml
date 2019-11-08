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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import ctypes
import math
import numpy as np
import warnings

from numba import cuda

from cuml.common.handle import Handle
from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from cuml import ForestInference
from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
from cuml.utils import get_cudf_column_ptr, get_dev_array_ptr, \
    input_to_dev_array, zeros
cimport cuml.common.handle
cimport cuml.common.cuda

cdef extern from "treelite/c_api.h":
    ctypedef void* ModelHandle
    ctypedef void* ModelBuilderHandle

cdef extern from "cuml/ensemble/randomforest.hpp" namespace "ML":
    cdef enum CRITERION:
        GINI,
        ENTROPY,
        MSE,
        MAE,
        CRITERION_END

cdef extern from "cuml/tree/decisiontree.hpp" namespace "ML::DecisionTree":
    cdef struct DecisionTreeParams:
        int max_depth
        int max_leaves
        float max_features
        int n_bins
        int split_algo
        int min_rows_per_node
        float min_impurity_decrease
        bool bootstrap_features
        bool quantile_per_tree
        CRITERION split_criterion

cdef extern from "cuml/ensemble/randomforest.hpp" namespace "ML":

    cdef enum RF_type:
        CLASSIFICATION,
        REGRESSION

    cdef struct RF_metrics:
        RF_type rf_type
        float accuracy
        double mean_abs_error
        double mean_squared_error
        double median_abs_error

    cdef struct RF_params:
        int n_trees
        bool bootstrap
        float rows_sample
        int seed
        pass

    cdef cppclass RandomForestMetaData[T, L]:
        void* trees
        RF_params rf_params

    cdef void fit(cumlHandle & handle,
                  RandomForestMetaData[float, float]*,
                  float*,
                  int,
                  int,
                  float*,
                  RF_params) except +

    cdef void fit(cumlHandle & handle,
                  RandomForestMetaData[double, double]*,
                  double*,
                  int,
                  int,
                  double*,
                  RF_params) except +

    cdef void predict(cumlHandle& handle,
                      RandomForestMetaData[float, float] *,
                      float*,
                      int,
                      int,
                      float*,
                      bool) except +

    cdef void predict(cumlHandle& handle,
                      RandomForestMetaData[double, double]*,
                      double*,
                      int,
                      int,
                      double*,
                      bool) except +

    cdef RF_metrics score(cumlHandle& handle,
                          RandomForestMetaData[float, float]*,
                          float*,
                          int,
                          float*,
                          bool) except +

    cdef RF_metrics score(cumlHandle& handle,
                          RandomForestMetaData[double, double]*,
                          double*,
                          int,
                          double*,
                          bool) except +

    cdef void build_treelite_forest(ModelHandle*,
                                    RandomForestMetaData[float, float]*,
                                    int,
                                    int) except +

    cdef void build_treelite_forest(ModelHandle*,
                                    RandomForestMetaData[double, double]*,
                                    int,
                                    int) except +

    cdef void print_rf_summary(RandomForestMetaData[float, float]*) except +
    cdef void print_rf_summary(RandomForestMetaData[double, double]*) except +

    cdef void print_rf_detailed(RandomForestMetaData[float, float]*) except +
    cdef void print_rf_detailed(RandomForestMetaData[double, double]*) except +

    cdef RF_params set_rf_class_obj(int,
                                    int,
                                    float,
                                    int,
                                    int,
                                    int,
                                    float,
                                    bool,
                                    bool,
                                    int,
                                    float,
                                    int,
                                    CRITERION,
                                    bool,
                                    int) except +


class RandomForestRegressor(Base):

    """
    Implements a Random Forest regressor model which fits multiple decision
    trees in an ensemble.
    Note that the underlying algorithm for tree node splits differs from that
    used in scikit-learn. By default, the cuML Random Forest uses a
    histogram-based algorithm to determine splits, rather than an exact
    count. You can tune the size of the histograms with the n_bins parameter.

    **Known Limitations**: This is an initial release of the cuML
    Random Forest code. It contains a few known limitations:

       * Inference/prediction takes place on the CPU. A GPU-based inference
         solution based on the forest inference library is planned for a
         near-future release.

       * Instances of RandomForestRegressor cannot be pickled currently.

    Examples
    ---------
    .. code-block:: python

            import numpy as np
            from cuml.test.utils import get_handle
            from cuml.ensemble import RandomForestRegressor as curfc
            from cuml.test.utils import get_handle
            X = np.asarray([[0,10],[0,20],[0,30],[0,40]], dtype=np.float32)
            y = np.asarray([0.0,1.0,2.0,3.0], dtype=np.float32)
            cuml_model = curfc(max_features=1.0, n_bins=8,
                               split_algo=0, min_rows_per_node=2,
                               n_estimators=40, accuracy_metric='mse')
            cuml_model.fit(X,y)
            cuml_score = cuml_model.score(X,y)
            print("MSE score of cuml : ", cuml_score)

    Output:

    .. code-block:: python

            MSE score of cuml :  0.1123437201231765

    Parameters
    -----------
    n_estimators : int (default = 10)
                   number of trees in the forest.
    handle : cuml.Handle
             If it is None, a new one is created just for this class.
    split_algo : int (default = 1)
                 0 for HIST, 1 for GLOBAL_QUANTILE
                 The type of algorithm to be used to create the trees.
                 HIST currently uses a slower tree-building algorithm
                 so GLOBAL_QUANTILE is recommended for most cases.
    split_criterion: int (default = 2)
                     The criterion used to split nodes.
                     0 for GINI, 1 for ENTROPY,
                     2 for MSE, or 3 for MAE
                     0 and 1 not valid for regression
    bootstrap : boolean (default = True)
                Control bootstrapping.
                If set, each tree in the forest is built
                on a bootstrapped sample with replacement.
                If false, sampling without replacement is done.
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
    max_features : int or float or string or None (default = 'auto')
                   Ratio of number of features (columns) to consider
                   per node split.
                   If int then max_features/n_features.
                   If float then max_features is a fraction.
                   If 'auto' then max_features=n_features which is 1.0.
                   If 'sqrt' then max_features=1/sqrt(n_features).
                   If 'log2' then max_features=log2(n_features)/n_features.
                   If None, then max_features=n_features which is 1.0.
    n_bins :  int (default = 8)
              Number of bins used by the split algorithm.
    min_rows_per_node : int or float (default = 2)
                        The minimum number of samples (rows) needed
                        to split a node.
                        If int then number of sample rows
                        If float the min_rows_per_sample*n_rows
    min_impurity_decrease : float (default = 0.0)
                            The minimum decrease in impurity required
                            for node to be split
    accuracy_metric : string (default = 'mse')
                      Decides the metric used to evaluate the performance
                      of the model.
                      for median of abs error : 'median_ae'
                      for mean of abs error : 'mean_ae'
                      for mean square error' : 'mse'
    quantile_per_tree : boolean (default = False)
                        Whether quantile is computed for individal trees in RF.
                        Only relevant for GLOBAL_QUANTILE split_algo.
    seed : int (default = -1)
           If int, then the seed is used by the models random number
           generator. It is used to reproduce the same result over multiple
           runs.

    """
    variables = ['n_estimators', 'max_depth', 'handle',
                 'max_features', 'n_bins',
                 'split_algo', 'split_criterion', 'min_rows_per_node',
                 'min_impurity_decrease',
                 'bootstrap', 'bootstrap_features',
                 'verbose', 'rows_sample',
                 'max_leaves', 'quantile_per_tree',
                 'accuracy_metric']

    def __init__(self, n_estimators=10, max_depth=16, handle=None,
                 max_features='auto', n_bins=8, n_streams=8,
                 split_algo=1, split_criterion=2,
                 bootstrap=True, bootstrap_features=False,
                 verbose=False, min_rows_per_node=2,
                 rows_sample=1.0, max_leaves=-1,
                 accuracy_metric='mse', min_samples_leaf=None,
                 min_weight_fraction_leaf=None, n_jobs=None,
                 max_leaf_nodes=None, min_impurity_decrease=0.0,
                 min_impurity_split=None, oob_score=None,
                 random_state=None, warm_start=None, class_weight=None,
                 quantile_per_tree=False, criterion=None, seed=-1):

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

        super(RandomForestRegressor, self).__init__(handle, verbose)

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
        self.accuracy_metric = accuracy_metric
        self.quantile_per_tree = quantile_per_tree
        self.n_streams = handle.getNumInternalStreams()
        self.seed = seed

        if ((seed != -1) and (n_streams != 1)):
            warnings.warn("Random seed only works with n_streams=1.")

        cdef RandomForestMetaData[float, float] *rf_forest = \
            new RandomForestMetaData[float, float]()
        self.rf_forest = <size_t> rf_forest
        cdef RandomForestMetaData[double, double] *rf_forest64 = \
            new RandomForestMetaData[double, double]()
        self.rf_forest64 = <size_t> rf_forest64
    """
    TODO:
        Add the preprocess and postprocess functions
        in the cython code to normalize the labels
    """
    def __getstate__(self):
        state = self.__dict__.copy()
        del state['handle']

        cdef size_t params_t = <size_t> self.rf_forest
        cdef  RandomForestMetaData[float, float] *rf_forest = \
            <RandomForestMetaData[float, float]*>params_t

        cdef size_t params_t64 = <size_t> self.rf_forest64
        cdef  RandomForestMetaData[double, double] *rf_forest64 = \
            <RandomForestMetaData[double, double]*>params_t64

        state['verbose'] = self.verbose

        if self.dtype == np.float32:
            state["rf_params"] = rf_forest.rf_params
            del state["rf_forest"]
        else:
            state["rf_params64"] = rf_forest64.rf_params
            del state["rf_forest64"]

        return state

    def __del__(self):
        cdef RandomForestMetaData[float, float]* rf_forest = \
            <RandomForestMetaData[float, float]*><size_t> self.rf_forest
        cdef RandomForestMetaData[double, double]* rf_forest64 = \
            <RandomForestMetaData[double, double]*><size_t> self.rf_forest64
        free(rf_forest)
        free(rf_forest64)

    def __setstate__(self, state):
        super(RandomForestRegressor, self).__init__(handle=None,
                                                    verbose=state['verbose'])
        cdef  RandomForestMetaData[float, float] *rf_forest = \
            new RandomForestMetaData[float, float]()

        cdef  RandomForestMetaData[double, double] *rf_forest64 = \
            new RandomForestMetaData[double, double]()

        if self.dtype == np.float32:
            rf_forest.rf_params = state["rf_params"]
            state["rf_forest"] = <size_t>rf_forest
        else:
            rf_forest64.rf_params = state["rf_params64"]
            state["rf_forest64"] = <size_t>rf_forest64

        self.__dict__.update(state)

    def _get_max_feat_val(self):
        if type(self.max_features) == int:
            return self.max_features/self.n_cols
        elif type(self.max_features) == float:
            return self.max_features
        elif self.max_features == 'sqrt':
            return 1/np.sqrt(self.n_cols)
        elif self.max_features == 'auto':
            return 1.0
        elif self.max_features == 'log2':
            return math.log2(self.n_cols)/self.n_cols
        else:
            raise ValueError("Wrong value passed in for max_features"
                             " please read the documentation")

    def fit(self, X, y):
        """
        Perform Random Forest Regression on the input data

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
        cdef RandomForestMetaData[float, float] *rf_forest = \
            <RandomForestMetaData[float, float]*><size_t> self.rf_forest
        cdef RandomForestMetaData[double, double] *rf_forest64 = \
            <RandomForestMetaData[double, double]*><size_t> self.rf_forest64

        y_m, y_ptr, _, _, _ = input_to_dev_array(y)

        X_m, X_ptr, n_rows, self.n_cols, self.dtype = \
            input_to_dev_array(X, order='F')

        if self.dtype == np.float64:
            warnings.warn("In order to run predict on the GPU convert"
                          " the data to float32")

        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()

        max_feature_val = self._get_max_feat_val()
        if type(self.min_rows_per_node) == float:
            self.min_rows_per_node = math.ceil(self.min_rows_per_node*n_rows)

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
                                     <int> self.seed,
                                     <CRITERION> self.split_criterion,
                                     <bool> self.quantile_per_tree,
                                     <int> self.n_streams)

        if self.dtype == np.float32:
            fit(handle_[0],
                rf_forest,
                <float*> X_ptr,
                <int> n_rows,
                <int> self.n_cols,
                <float*> y_ptr,
                rf_params)

        else:
            rf_params64 = rf_params
            fit(handle_[0],
                rf_forest64,
                <double*> X_ptr,
                <int> n_rows,
                <int> self.n_cols,
                <double*> y_ptr,
                rf_params64)
        # make sure that the `fit` is complete before the following delete
        # call happens
        self.handle.sync()
        del(X_m)
        del(y_m)
        return self

    def _predict_model_on_gpu(self, X,
                              output_class,
                              algo):
        _, _, n_rows, n_cols, X_dtype = \
            input_to_dev_array(X, order='C', check_dtype=self.dtype,
                               check_cols=self.n_cols)

        # task category = 1 for regression
        treelite_model = self._get_treelite(num_features=n_cols,
                                            task_category=1)

        fil_model = ForestInference()
        tl_to_fil_model = \
            fil_model.load_from_randomforest(treelite_model,
                                             output_class=False,
                                             algo='BATCH_TREE_REORG')
        preds = tl_to_fil_model.predict(X)
        return preds

    def _predict_model_on_cpu(self, X):
        cdef uintptr_t X_ptr
        X_m, X_ptr, n_rows, n_cols, _ = \
            input_to_dev_array(X, order='C', check_dtype=self.dtype,
                               check_cols=self.n_cols)
        if n_cols != self.n_cols:
            raise ValueError("The number of columns/features in the training"
                             " and test data should be the same ")

        preds = np.zeros(n_rows, dtype=self.dtype)
        cdef uintptr_t preds_ptr
        preds_m, preds_ptr, _, _, _ = \
            input_to_dev_array(preds)
        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()

        cdef RandomForestMetaData[float, float] *rf_forest = \
            <RandomForestMetaData[float, float]*><size_t> self.rf_forest

        cdef RandomForestMetaData[double, double] *rf_forest64 = \
            <RandomForestMetaData[double, double]*><size_t> self.rf_forest64
        if self.dtype == np.float32:
            predict(handle_[0],
                    rf_forest,
                    <float*> X_ptr,
                    <int> n_rows,
                    <int> n_cols,
                    <float*> preds_ptr,
                    <bool> self.verbose)

        elif self.dtype == np.float64:
            predict(handle_[0],
                    rf_forest64,
                    <double*> X_ptr,
                    <int> n_rows,
                    <int> n_cols,
                    <double*> preds_ptr,
                    <bool> self.verbose)
        else:
            raise TypeError("supports only float32 and float64 input,"
                            " but input of type '%s' passed."
                            % (str(self.dtype)))

        self.handle.sync()
        # synchronous w/o a stream
        preds = preds_m.copy_to_host()
        del(X_m)
        del(preds_m)
        return preds

    def predict(self, X, predict_model="GPU",
                output_class=False,
                algo='BATCH_TREE_REORG'):
        """
        Predicts the labels for X.
        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy
        predict_model : String
                        "GPU" if prediction should be carried out on the GPU
                        "CPU" or None if prediction should be carried out
                        on the CPU
        output_class: boolean
                      This is optional and required only while performing the
                      predict operation on the GPU.
                      If true, return a 1 or 0 depending on whether the raw
                      prediction exceeds the threshold. If False, just return
                      the raw prediction.
        algo : string name of the algo from (from algo_t enum)
               This is optional and required only while performing the
               predict operation on the GPU.
               'NAIVE' - simple inference using shared memory
               'TREE_REORG' - similar to naive but trees rearranged to be more
                              coalescing-friendly
               'BATCH_TREE_REORG' - similar to TREE_REORG but predicting
                                    multiple rows per thread block
        Returns
        ----------
        y: NumPy
           Dense vector (int) of shape (n_samples, 1)
        """
        if self.dtype == np.float64:
            raise TypeError("GPU predict model only accepts float32 dtype"
                            " as input, convert the data to float32 or "
                            "use the CPU predict with `predict_model='CPU'`.")

        elif predict_model == "GPU":
            preds = self._predict_model_on_gpu(X,
                                               output_class,
                                               algo)
        else:
            preds = self._predict_model_on_cpu(X)
        return preds

    def score(self, X, y, algo='BATCH_TREE_REORG'):
        """
        Calculates the accuracy metric score of the model for X.
        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy
        y: NumPy
           Dense vector (int) of shape (n_samples, 1)
        algo : string name of the algo from (from algo_t enum)
               This is optional and required only while performing the
               predict operation on the GPU.
               'NAIVE' - simple inference using shared memory
               'TREE_REORG' - similar to naive but trees rearranged to be more
                              coalescing-friendly
               'BATCH_TREE_REORG' - similar to TREE_REORG but predicting
                                    multiple rows per thread block
        Returns
        ----------
        mean_square_error : float or
        median_abs_error : float or
        mean_abs_error : float
        """
        cdef uintptr_t X_ptr, y_ptr
        y_m, y_ptr, n_rows, _, _ = \
            input_to_dev_array(y, check_dtype=self.dtype)

        preds = self._predict_model_on_gpu(X, output_class=False,
                                           algo=algo)
        cdef uintptr_t preds_ptr
        preds_m, preds_ptr, _, _, _ = \
            input_to_dev_array(preds)

        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()

        cdef RandomForestMetaData[float, float] *rf_forest = \
            <RandomForestMetaData[float, float]*><size_t> self.rf_forest

        cdef RandomForestMetaData[double, double] *rf_forest64 = \
            <RandomForestMetaData[double, double]*><size_t> self.rf_forest64

        if self.dtype == np.float32:
            self.temp_stats = score(handle_[0],
                                    rf_forest,
                                    <float*> y_ptr,
                                    <int> n_rows,
                                    <float*> preds_ptr,
                                    <bool> self.verbose)

        elif self.dtype == np.float64:
            self.temp_stats = score(handle_[0],
                                    rf_forest64,
                                    <double*> y_ptr,
                                    <int> n_rows,
                                    <double*> preds_ptr,
                                    <bool> self.verbose)

        if self.accuracy_metric == 'median_ae':
            stats = self.temp_stats['median_abs_error']
        if self.accuracy_metric == 'mean_ae':
            stats = self.temp_stats['mean_abs_error']
        else:
            stats = self.temp_stats['mean_squared_error']

        self.handle.sync()
        del(y_m)
        del(preds_m)
        return stats

    def get_params(self, deep=True):
        """
        Returns the value of all parameters
        required to configure this estimator as a dictionary.
        Parameters
        -----------
        deep : boolean (default = True)
        """

        params = dict()
        for key in RandomForestRegressor.variables:
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
        self.__init__()
        if not params:
            return self
        for key, value in params.items():
            if key not in RandomForestRegressor.variables:
                raise ValueError('Invalid parameter for estimator')
            else:
                setattr(self, key, value)

        return self

    def print_summary(self):
        """
        prints the summary of the forest used to train and test the model
        """
        cdef RandomForestMetaData[float, float] *rf_forest = \
            <RandomForestMetaData[float, float]*><size_t> self.rf_forest

        cdef RandomForestMetaData[double, double] *rf_forest64 = \
            <RandomForestMetaData[double, double]*><size_t> self.rf_forest64

        if self.dtype == np.float64:
            print_rf_summary(rf_forest64)
        else:
            print_rf_summary(rf_forest)

    def print_detailed(self):
        """
        prints the detailed information about the forest used to
        train and test the Random Forest model
        """
        cdef RandomForestMetaData[float, float] *rf_forest = \
            <RandomForestMetaData[float, float]*><size_t> self.rf_forest

        cdef RandomForestMetaData[double, double] *rf_forest64 = \
            <RandomForestMetaData[double, double]*><size_t> self.rf_forest64

        if self.dtype == np.float64:
            print_rf_detailed(rf_forest64)
        else:
            print_rf_detailed(rf_forest)

    def _get_treelite(self, num_features,
                      task_category=1, model=None):

        cdef ModelHandle cuml_model_ptr
        cdef RandomForestMetaData[float, float] *rf_forest = \
            <RandomForestMetaData[float, float]*><size_t> self.rf_forest

        cdef RandomForestMetaData[double, double] *rf_forest64 = \
            <RandomForestMetaData[double, double]*><size_t> self.rf_forest64

        cdef ModelBuilderHandle tl_model_ptr
        if self.dtype == np.float32:
            build_treelite_forest(& cuml_model_ptr,
                                  rf_forest,
                                  <int> num_features,
                                  <int> task_category)

        else:
            build_treelite_forest(& cuml_model_ptr,
                                  rf_forest64,
                                  <int> num_features,
                                  <int> task_category)
        mod_ptr = <size_t> cuml_model_ptr

        return ctypes.c_void_p(mod_ptr).value

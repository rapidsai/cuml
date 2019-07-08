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
import numpy as np
import warnings
from numba import cuda

from libcpp cimport bool
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
from cuml.utils import get_cudf_column_ptr, get_dev_array_ptr, \
    input_to_dev_array, zeros
cimport cuml.common.handle
cimport cuml.common.cuda

cdef extern from "randomforest/randomforest.h" namespace "ML":

    cdef struct RF_metrics:
        float accuracy
        double mean_abs_error
        double mean_squared_error
        double median_abs_error

        RF_metrics(double, double,
                   double)

    cdef enum RF_type:
        CLASSIFICATION,
        REGRESSION

    cdef enum CRITERION:
        GINI,
        ENTROPY,
        MSE,
        MAE,
        CRITERION_END

    cdef struct RF_params:
        pass

    cdef cppclass rf:
        rf(RF_params, int)

    cdef cppclass rfRegressor[T]:
        rfRegressor(RF_params) except +

    cdef void fit(cumlHandle& handle,
                  rfRegressor[float]*,
                  float*,
                  int,
                  int,
                  float*) except +

    cdef void fit(cumlHandle& handle,
                  rfRegressor[double]*,
                  double*,
                  int,
                  int,
                  double*) except +

    cdef void predict(cumlHandle& handle,
                      rfRegressor[float]*,
                      float*,
                      int,
                      int,
                      float*,
                      bool) except +

    cdef void predict(cumlHandle& handle,
                      rfRegressor[double]*,
                      double*,
                      int,
                      int,
                      double*,
                      bool) except +

    cdef RF_metrics score(cumlHandle& handle,
                          rfRegressor[float]*,
                          float*,
                          float*,
                          int,
                          int,
                          float*,
                          bool)
    cdef RF_metrics score(cumlHandle& handle,
                          rfRegressor[double]*,
                          double*,
                          double*,
                          int,
                          int,
                          double*,
                          bool)

    cdef RF_params set_rf_class_obj(int, int, float,
                                    int, int, int,
                                    bool, bool, int, float, CRITERION,
                                    bool) except +

cdef class RandomForest_impl():

    cpdef object handle
    cdef rfRegressor[float] *rf_regressor32
    cdef rfRegressor[double] *rf_regressor64
    cdef RF_metrics temp_stats
    cdef uintptr_t preds_ptr
    cdef object n_estimators
    cdef object max_depth
    cdef object max_features
    cdef object n_bins
    cdef object split_algo
    cdef object quantile_per_tree
    cdef object split_criterion
    cdef object criterion
    cdef object min_rows_per_node
    cdef object bootstrap
    cdef object bootstrap_features
    cdef object verbose
    cdef object n_cols
    cdef object rows_sample
    cdef object max_leaves
    cdef object gdf_datatype
    cdef object accuracy_metric
    cdef object dtype

    def __cinit__(self, n_estimators=10, max_depth=-1, handle=None,
                  max_features=1.0, n_bins=8,
                  split_algo=0, split_criterion=2, min_rows_per_node=2,
                  bootstrap=True, bootstrap_features=False,
                  verbose=False, rows_sample=1.0,
                  max_leaves=-1, accuracy_metric='mse',
                  quantile_per_tree=False):

        self.handle = handle
        self.accuracy_metric = accuracy_metric
        self.split_algo = split_algo
        self.criterion = split_criterion
        self.min_rows_per_node = min_rows_per_node
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
        self.rf_regressor32 = NULL
        self.rf_regressor64 = NULL
        self.n_cols = None

    def _get_type(self):
        if self.criterion == 0:
            return GINI
        elif self.criterion == 1:
            return ENTROPY
        elif self.criterion == 2:
            return MSE
        elif self.criterion == 3:
            return MAE
        else:
            return CRITERION_END

    def fit(self, X, y):

        cdef uintptr_t X_ptr, y_ptr

        if self.rf_regressor32 != NULL:
            del self.rf_regressor32

        if self.rf_regressor64 != NULL:
            del self.rf_regressor64

        y_m, y_ptr, _, _, y_dtype = input_to_dev_array(y)

        X_m, X_ptr, n_rows, self.n_cols, self.dtype = \
            input_to_dev_array(X, order='F')

        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()

        self.split_criterion = self._get_type()
        rf_param = set_rf_class_obj(<int> self.max_depth,
                                    <int> self.max_leaves,
                                    <float> self.max_features,
                                    <int> self.n_bins,
                                    <int> self.split_algo,
                                    <int> self.min_rows_per_node,
                                    <bool> self.bootstrap_features,
                                    <bool> self.bootstrap,
                                    <int> self.n_estimators,
                                    <int> self.rows_sample,
                                    <CRITERION> self.split_criterion,
                                    <bool> self.quantile_per_tree)

        self.rf_regressor32 = new \
            rfRegressor[float](rf_param)
        self.rf_regressor64 = new \
            rfRegressor[double](rf_param)

        if self.dtype == np.float32:
            fit(handle_[0],
                self.rf_regressor32,
                <float*> X_ptr,
                <int> n_rows,
                <int> self.n_cols,
                <float*> y_ptr)
        else:
            fit(handle_[0],
                self.rf_regressor64,
                <double*> X_ptr,
                <int> n_rows,
                <int> self.n_cols,
                <double*> y_ptr)

        # make sure that the `fit` is complete before the following delete
        # call happens
        self.handle.sync()
        del(X_m)
        del(y_m)
        return self

    def predict(self, X):

        cdef uintptr_t X_ptr
        # row major format
        X_m, X_ptr, n_rows, n_cols, X_dtype = \
            input_to_dev_array(X, order='C')
        if n_cols != self.n_cols:
            raise ValueError(" The number of columns/features in the training"
                             " and test data should be the same ")
        if X_dtype != self.dtype:
            raise ValueError(" The datatype of the training data is different"
                             " from the datatype of the testing data")
        preds = np.zeros(n_rows, dtype=self.dtype)
        cdef uintptr_t preds_ptr
        preds_m, preds_ptr, _, _, _ = \
            input_to_dev_array(preds)
        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()

        if self.dtype == np.float32:
            predict(handle_[0],
                    self.rf_regressor32,
                    <float*> X_ptr,
                    <int> n_rows,
                    <int> n_cols,
                    <float*> preds_ptr,
                    <bool> self.verbose)

        elif self.dtype == np.float64:
            predict(handle_[0],
                    self.rf_regressor64,
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

    def score(self, X, y):

        cdef uintptr_t X_ptr, y_ptr
        X_m, X_ptr, n_rows, n_cols, X_dtype = \
            input_to_dev_array(X, order='C')
        y_m, y_ptr, _, _, _ = input_to_dev_array(y)

        if n_cols != self.n_cols:
            raise ValueError(" The number of columns/features in the training"
                             " and test data should be the same ")
        if X_dtype != self.dtype:
            raise ValueError(" The datatype of the training data is different"
                             " from the datatype of the testing data")

        preds = np.zeros(n_rows,
                         dtype=self.dtype)
        cdef uintptr_t preds_ptr
        preds_m, preds_ptr, _, _, _ = \
            input_to_dev_array(preds)

        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()

        if self.dtype == np.float32:
            self.temp_stats = score(handle_[0],
                                    self.rf_regressor32,
                                    <float*> X_ptr,
                                    <float*> y_ptr,
                                    <int> n_rows,
                                    <int> n_cols,
                                    <float*> preds_ptr,
                                    <bool> self.verbose)

        elif self.dtype == np.float64:
            self.temp_stats = score(handle_[0],
                                    self.rf_regressor64,
                                    <double*> X_ptr,
                                    <double*> y_ptr,
                                    <int> n_rows,
                                    <int> n_cols,
                                    <double*> preds_ptr,
                                    <bool> self.verbose)

        if self.accuracy_metric == 'median_ae':
            stats = self.temp_stats.median_abs_error
        if self.accuracy_metric == 'mean_ae':
            stats = self.temp_stats.mean_abs_error
        else:
            stats = self.temp_stats.mean_squared_error

        self.handle.sync()
        del(X_m)
        del(y_m)
        del(preds_m)
        return stats


class RandomForestRegressor(Base):

    """
    Implements a Random Forest classifier model which fits multiple decision
    tree classifiers in an ensemble.
    Note that the underlying algorithm for tree node splits differs from that
    used in scikit-learn. By default, the cuML Random Forest uses a
    histogram-based algorithms to determine splits, rather than an exact
    count. You can tune the size of the histograms with the n_bins parameter.

    The instances of RandomForestRegressor cannot be pickled currently.

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
    split_algo : 0 for HIST, 1 for GLOBAL_QUANTILE and 2 for SPLIT_ALGO_END
                 (default = 0)
                 The type of algorithm to be used to create the trees.
    split_criterion: The criterion used to split nodes.
                     0 for GINI, 1 for ENTROPY,
                     2 for MAE, 3 for MSE and 4 for CRITERION_END.
                     0 and 1 not valid for classification
                     (default = 2)
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
    max_depth : int (default = -1)
                Maximum tree depth. Unlimited (i.e, until leaves are pure),
                if -1.
    max_leaves : int (default = -1)
                 Maximum leaf nodes per tree. Soft constraint. Unlimited,
                 if -1.
    max_features : float (default = 1.0)
                   Ratio of number of features (columns) to consider
                   per node split.
    n_bins :  int (default = 8)
              Number of bins used by the split algorithm.
    min_rows_per_node : int (default = 2)
                        The minimum number of samples (rows) needed
                        to split a node.
    accuracy_metric : string (default = 'mse')
                      Decides the metric used to evaluate the performance
                      of the model.
                      'median'

    """

    variables = ['n_estimators', 'max_depth', 'handle',
                 'max_features', 'n_bins',
                 'split_algo', 'split_criterion', 'min_rows_per_node',
                 'bootstrap', 'bootstrap_features',
                 'verbose', 'rows_sample',
                 'max_leaves', 'quantile_per_tree',
                 'accuracy_metric']

    def __init__(self, n_estimators=10, max_depth=-1, handle=None,
                 max_features=1.0, n_bins=8,
                 split_algo=0, split_criterion=2,
                 bootstrap=True, bootstrap_features=False,
                 verbose=False, min_rows_per_node=2,
                 rows_sample=1.0, max_leaves=-1,
                 accuracy_metric='mse', min_samples_leaf=None,
                 min_weight_fraction_leaf=None, n_jobs=None,
                 max_leaf_nodes=None, min_impurity_decrease=None,
                 min_impurity_split=None, oob_score=None,
                 random_state=None, warm_start=None, class_weight=None,
                 quantile_per_tree=False, criterion=None):

        sklearn_params = {"criterion": criterion,
                          "min_samples_leaf": min_samples_leaf,
                          "min_weight_fraction_leaf": min_weight_fraction_leaf,
                          "max_leaf_nodes": max_leaf_nodes,
                          "min_impurity_decrease": min_impurity_decrease,
                          "min_impurity_split": min_impurity_split,
                          "oob_score": oob_score, "n_jobs": n_jobs,
                          "random_state": random_state,
                          "warm_start": warm_start,
                          "class_weight": class_weight}

        for key, vals in sklearn_params.items():
            if vals is not None:
                raise TypeError(" The sklearn variable ", key,
                                " is not supported in cuML,"
                                " please read the cuML documentation for"
                                " more information")

        super(RandomForestRegressor, self).__init__(handle, verbose)

        self.split_algo = split_algo
        self.split_criterion = split_criterion
        self.min_rows_per_node = min_rows_per_node
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
        self._impl = RandomForest_impl(n_estimators, max_depth, self.handle,
                                       max_features, n_bins,
                                       split_algo, split_criterion,
                                       min_rows_per_node,
                                       bootstrap, bootstrap_features,
                                       verbose,
                                       rows_sample, max_leaves,
                                       accuracy_metric,
                                       quantile_per_tree)

    def fit(self, X, y):
        """
        Perform Random Forest Regression on the input data. The input array X
        and y should have the same datatype.

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy
        y : array-like (device or host) shape = (n_samples, 1)
            Dense vector (int) of shape (n_samples, 1).
            Acceptable formats: NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy
        """

        return self._impl.fit(X, y)

    def predict(self, X):
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
        y: NumPy
           Dense vector (int) of shape (n_samples, 1)

        """

        return self._impl.predict(X)

    def score(self, X, y):
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

        Returns
        ----------
        mean_square_error : float or
        median_abs_error : float or
        mean_abs_error : float
        """
        return self._impl.score(X, y)

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
        if not params:
            return self
        for key, value in params.items():
            if key not in RandomForestRegressor.variables:
                raise ValueError('Invalid parameter for estimator')
            else:
                setattr(self, key, value)

        self._impl = RandomForest_impl(self.n_estimators, self.max_depth,
                                       self.handle, self.max_features,
                                       self.n_bins, self.split_algo,
                                       self.split_criterion,
                                       self.min_rows_per_node, self.bootstrap,
                                       self.bootstrap_features, self.verbose,
                                       self.rows_sample, self.max_leaves,
                                       self.accuracy_metric,
                                       self.quantile_per_tree)

        return self

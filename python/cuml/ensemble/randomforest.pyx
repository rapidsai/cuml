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


cdef extern from "randomforest/randomforest.hpp" namespace "ML":

    cdef struct RF_metrics:
        float accuracy
        double mean_abs_error
        double mean_squared_error
        double median_abs_error

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
        int n_trees
        bool bootstrap
        float rows_sample
        # tree params placeholder
        pass


    cdef cppclass RandomForestMetaData[T, L]:
        #DecisionTree::TreeMetaDataNode<T, L>* trees;
        void* trees
        RF_params rf_params


    # Random Forest Classifier
    cdef void fit(cumlHandle & handle,
                  RandomForestMetaData[float, int] *,
                  float *,
                  int,
                  int,
                  int *,
                  int,
                  RF_params) except +

    cdef void fit(cumlHandle & handle,
                  RandomForestMetaData[double, int] *,
                  double *,
                  int,
                  int,
                  int *,
                  int,
                  RF_params) except +

    cdef void predict(cumlHandle& handle,
                      RandomForestMetaData[float, int] *,
                      float *,
                      int,
                      int,
                      int *,
                      bool) except +

    cdef void predict(cumlHandle& handle,
                      RandomForestMetaData[double, int] *,
                      double *,
                      int,
                      int,
                      int *,
                      bool) except +

    cdef RF_metrics score(cumlHandle& handle,
                          RandomForestMetaData[float, int] *,
                          float *, int *,
                          int, int, int *, bool) except +

    cdef RF_metrics score(cumlHandle& handle,
                          RandomForestMetaData[double, int] *,
                          double *, int *,
                          int, int, int *, bool) except +

    cdef RF_params set_rf_class_obj(int, int, float,
                                    int, int, int,
                                    bool, bool, int, float, CRITERION,
                                    bool) except +

    # Random Forest Regressor
    cdef void fit(cumlHandle & handle,
                  RandomForestMetaData[float, float] *,
                  float *,
                  int,
                  int,
                  float *,
                  RF_params) except +

    cdef void fit(cumlHandle & handle,
                  RandomForestMetaData[double, double] *,
                  double *,
                  int,
                  int,
                  double *,
                  RF_params) except +

    cdef void predict(cumlHandle& handle,
                      RandomForestMetaData[float, float] *,
                      float *,
                      int,
                      int,
                      float *,
                      bool) except +

    cdef void predict(cumlHandle& handle,
                      RandomForestMetaData[double, double] *,
                      double *,
                      int,
                      int,
                      double *,
                      bool) except +

    cdef RF_metrics score(cumlHandle& handle,
                          RandomForestMetaData[float, float] *,
                          float *, float *,
                          int, int, float *, bool) except +

    cdef RF_metrics score(cumlHandle& handle,
                          RandomForestMetaData[double, double] *,
                          double *, double *,
                          int, int, double *, bool) except +

    cdef RF_params set_rf_class_obj(int, int, float,
                                    int, int, int,
                                    bool, bool, int, float, CRITERION,
                                    bool) except +

cdef class RandomForest_impl():

    cpdef object handle
    cdef RandomForestMetaData[float, int] *rf_forest32
    cdef RandomForestMetaData[double, int] *rf_forest64
    cdef uintptr_t preds_ptr
    cdef object n_estimators
    cdef object max_depth
    cdef object max_features
    cdef object n_bins
    cdef object split_algo
    cdef object split_criterion
    cdef object min_rows_per_node
    cdef object bootstrap
    cdef object bootstrap_features
    cdef object type_model
    cdef object verbose
    cdef object n_cols
    cdef object rows_sample
    cdef object max_leaves
    cdef object quantile_per_tree
    cdef object gdf_datatype
    cdef RF_metrics stats
    cdef object dtype

    def __cinit__(self, n_estimators=10, max_depth=-1, handle=None,
                  max_features=1.0, n_bins=8,
                  split_algo=0, split_criterion=0, min_rows_per_node=2,
                  bootstrap=True, bootstrap_features=False,
                  type_model="classifier", verbose=False,
                  rows_sample=1.0, max_leaves=-1, quantile_per_tree=False,
                  gdf_datatype=None):

        self.handle = handle
        self.split_algo = split_algo
        self.split_criterion = split_criterion
        self.min_rows_per_node = min_rows_per_node
        self.bootstrap_features = bootstrap_features
        self.rows_sample = rows_sample
        self.max_leaves = max_leaves
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.type_model = self._get_type(type_model)
        self.quantile_per_tree = quantile_per_tree
        self.bootstrap = bootstrap
        self.verbose = verbose
        self.n_bins = n_bins
        self.rf_forest32 = NULL
        self.rf_forest64 = NULL
        self.n_cols = None

    def _get_type(self, type_model):
        if type_model == "classifier":
            self.type_model = 1
        else:
            self.type_model = 0

    """
    TODO:
        Add the preprocess and postprocess functions
        in the cython code to normalize the labels
    """

    def fit(self, X, y):

        cdef uintptr_t X_ptr, y_ptr

        #if self.rf_forest32 != NULL:
        #    del self.rf_forest32

        #if self.rf_forest64 != NULL:
        #    del self.rf_forest64

        y_m, y_ptr, _, _, y_dtype = input_to_dev_array(y)

        if y_dtype != np.int32:
            raise TypeError("The labels need to have dtype = np.int32")

        X_m, X_ptr, n_rows, self.n_cols, self.dtype = \
            input_to_dev_array(X, order='F')

        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()

        try:
            import cupy as cp
            unique_labels = cp.unique(y_m)
        except ImportError:
            warnings.warn("Using NumPy for number of class detection,"
                          "install CuPy for faster processing.")
            if isinstance(y, np.ndarray):
                unique_labels = np.unique(y)
            else:
                unique_labels = np.unique(y_m.copy_to_host())

        num_unique_labels = (unique_labels).__len__()
        for i in range(num_unique_labels):
            if i not in unique_labels:
                raise ValueError("The labels need "
                                 "to be from 0 to num_unique_label values")

        rf_params = set_rf_class_obj(<int> self.max_depth,
                                    <int> self.max_leaves,
                                    <float> self.max_features,
                                    <int> self.n_bins,
                                    <int> self.split_algo,
                                    <int> self.min_rows_per_node,
                                    <bool> self.bootstrap_features,
                                    <bool> self.bootstrap,
                                    <int> self.n_estimators,
                                    <float> self.rows_sample,
                                    <CRITERION> self.split_criterion,
                                    <bool> self.quantile_per_tree)

        self.rf_forest32 = new RandomForestMetaData[float, int]()
        self.rf_forest64 = new RandomForestMetaData[double, int]()

        if self.dtype == np.float32:
            fit(handle_[0],
                self.rf_forest32,
                <float*> X_ptr,
                <int> n_rows,
                <int> self.n_cols,
                <int*> y_ptr,
                <int> num_unique_labels,
                rf_params)
        else:
            fit(handle_[0],
                self.rf_forest64,
                <double*> X_ptr,
                <int> n_rows,
                <int> self.n_cols,
                <int*> y_ptr,
                <int> num_unique_labels,
                rf_params)
        # make sure that the `fit` is complete before the following delete
        # call happens
        self.handle.sync()
        del(X_m)
        del(y_m)
        return self

    def predict(self, X):

        cdef uintptr_t X_ptr
        # row major format
        X_m, X_ptr, n_rows, n_cols, _ = \
            input_to_dev_array(X, order='C')
        if n_cols != self.n_cols:
            raise ValueError("The number of columns/features in the training"
                             " and test data should be the same ")
        if X.dtype != self.dtype:
            raise ValueError("The datatype of the training data is different"
                             " from the datatype of the testing data")

        preds = np.zeros(n_rows, dtype=np.int32)
        cdef uintptr_t preds_ptr
        preds_m, preds_ptr, _, _, _ = \
            input_to_dev_array(preds)
        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()

        if self.dtype == np.float32:
            predict(handle_[0],
                    self.rf_forest32,
                    <float*> X_ptr,
                    <int> n_rows,
                    <int> n_cols,
                    <int*> preds_ptr,
                    <bool> self.verbose)

        elif self.dtype == np.float64:
            predict(handle_[0],
                    self.rf_forest64,
                    <double*> X_ptr,
                    <int> n_rows,
                    <int> n_cols,
                    <int*> preds_ptr,
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
        X_m, X_ptr, n_rows, n_cols, _ = \
            input_to_dev_array(X, order='C')
        y_m, y_ptr, _, _, _ = input_to_dev_array(y)

        if n_cols != self.n_cols:
            raise ValueError("The number of columns/features in the training"
                             " and test data should be the same ")
        if y.dtype != np.int32:
            raise TypeError("The labels need to have dtype = np.int32")

        if X.dtype != self.dtype:
            raise ValueError("The datatype of the training data is different"
                             " from the datatype of the testing data")

        preds = np.zeros(n_rows,
                         dtype=np.int32)
        cdef uintptr_t preds_ptr
        preds_m, preds_ptr, _, _, _ = \
            input_to_dev_array(preds)

        cdef cumlHandle* handle_ =\
            <cumlHandle*><size_t>self.handle.getHandle()

        if self.dtype == np.float32:
            self.stats = score(handle_[0],
                               self.rf_forest32,
                               <float*> X_ptr,
                               <int*> y_ptr,
                               <int> n_rows,
                               <int> n_cols,
                               <int*> preds_ptr,
                               <bool> self.verbose)

        elif self.dtype == np.float64:
            self.stats = score(handle_[0],
                               self.rf_forest64,
                               <double*> X_ptr,
                               <int*> y_ptr,
                               <int> n_rows,
                               <int> n_cols,
                               <int*> preds_ptr,
                               <bool> self.verbose)

        self.handle.sync()
        del(X_m)
        del(y_m)
        del(preds_m)
        return self.stats.accuracy


class RandomForestClassifier(Base):
    """
    Implements a Random Forest classifier model which fits multiple decision
    tree classifiers in an ensemble.

    Note that the underlying algorithm for tree node splits differs from that
    used in scikit-learn. By default, the cuML Random Forest uses a
    histogram-based algorithms to determine splits, rather than an exact
    count. You can tune the size of the histograms with the n_bins parameter.

    **Known Limitations**: This is an initial preview release of the cuML
    Random Forest code. It contains a number of known
    limitations:

       * Only classification is supported. Regression support is planned for
         the next release.

       * The implementation relies on limited CUDA shared memory for scratch
         space, so models with a very large number of features or bins will
         generate a memory limit exception. This limitation will be lifted in
         the next release.

       * Inference/prediction takes place on the CPU. A GPU-based inference
         solution is planned for a near-future release release.

       * Instances of RandomForestClassifier cannot be pickled currently.

    The code is under heavy development, so users who need these features may
    wish to pull from nightly builds of cuML. (See https://rapids.ai/start.html
    for instructions to download nightly packages via conda.)

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
    n_estimators : int (default = 10)
                   number of trees in the forest.
    handle : cuml.Handle
             If it is None, a new one is created just for this class.
    split_criterion: The criterion used to split nodes.
                     0 for GINI, 1 for ENTROPY, 4 for CRITERION_END.
                     2 and 3 not valid for classification
                     (default = 0)
    split_algo : 0 for HIST and 1 for GLOBAL_QUANTILE
                 (default = 0)
                 the algorithm to determine how nodes are split in the tree.
    split_criterion: The criterion used to split nodes.
                     0 for GINI, 1 for ENTROPY, 4 for CRITERION_END.
                     2 and 3 not valid for classification
                     (default = 0)
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
    quantile_per_tree : boolean (default = False)
                        Whether quantile is computed for individal trees in RF.
                        Only relevant for GLOBAL_QUANTILE split_algo.

    """

    variables = ['n_estimators', 'max_depth', 'handle',
                 'max_features', 'n_bins',
                 'split_algo', 'split_criterion', 'min_rows_per_node',
                 'bootstrap', 'bootstrap_features',
                 'verbose', 'rows_sample',
                 'max_leaves', 'quantile_per_tree']

    def __init__(self, n_estimators=10, max_depth=-1, handle=None,
                 max_features=1.0, n_bins=8,
                 split_algo=0, split_criterion=0, min_rows_per_node=2,
                 bootstrap=True, bootstrap_features=False,
                 type_model="classifier", verbose=False,
                 rows_sample=1.0, max_leaves=-1, quantile_per_tree=False,
                 gdf_datatype=None, criterion=None,
                 min_samples_leaf=None, min_weight_fraction_leaf=None,
                 max_leaf_nodes=None, min_impurity_decrease=None,
                 min_impurity_split=None, oob_score=None, n_jobs=None,
                 random_state=None, warm_start=None, class_weight=None):

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
                raise TypeError("The sklearn variable ", key,
                                " is not supported in cuML,"
                                " please read the cuML documentation for"
                                " more information")

        super(RandomForestClassifier, self).__init__(handle, verbose)

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
        self.quantile_per_tree = quantile_per_tree

        self._impl = RandomForest_impl(n_estimators, max_depth, self.handle,
                                       max_features, n_bins,
                                       split_algo, split_criterion,
                                       min_rows_per_node,
                                       bootstrap, bootstrap_features,
                                       type_model, verbose,
                                       rows_sample, max_leaves,
                                       quantile_per_tree,
                                       gdf_datatype)

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

        return self._impl.fit(X, y)

    def predict(self, X):
        """
        Predicts the labels for X.

        Parameters
        ----------
        X : array-like (host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: NumPy ndarray, Numba device
            ndarray

        Returns
        ----------
        y: NumPy
           Dense vector (int) of shape (n_samples, 1)

        """

        return self._impl.predict(X)

    def score(self, X, y):
        """
        Predicts the accuracy of the model for X.

        Parameters
        ----------
        X : array-like (host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: NumPy ndarray, Numba device
            ndarray

        y: NumPy
           Dense vector (int) of shape (n_samples, 1)

        Returns
        ----------
        accuracy : float
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
        for key in RandomForestClassifier.variables:
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

        return self

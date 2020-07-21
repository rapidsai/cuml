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

import numpy as np
import rmm
import warnings

import cuml.common.logger as logger

from cuml import ForestInference
from cuml.common.array import CumlArray

from cuml.common.base import RegressorMixin
from cuml.common.handle import Handle
from cuml.common import input_to_cuml_array, rmm_cupy_ary

from cuml.ensemble.randomforest_common import BaseRandomForestModel
from cuml.ensemble.randomforest_common import _obtain_fil_model
from cuml.ensemble.randomforest_shared cimport *

from cuml.fil.fil import TreeliteModel

from cython.operator cimport dereference as deref

from libcpp cimport bool
from libcpp.vector cimport vector
from libc.stdint cimport uintptr_t
from libc.stdlib cimport calloc, malloc, free

from numba import cuda

from cuml.common.handle cimport cumlHandle
cimport cuml.common.handle
cimport cuml.common.cuda

cimport cython


cdef extern from "cuml/ensemble/randomforest.hpp" namespace "ML":

    cdef void fit(cumlHandle& handle,
                  RandomForestMetaData[float, float]*,
                  float*,
                  int,
                  int,
                  float*,
                  RF_params,
                  int) except +

    cdef void fit(cumlHandle& handle,
                  RandomForestMetaData[double, double]*,
                  double*,
                  int,
                  int,
                  double*,
                  RF_params,
                  int) except +

    cdef void predict(cumlHandle& handle,
                      RandomForestMetaData[float, float] *,
                      float*,
                      int,
                      int,
                      float*,
                      int) except +

    cdef void predict(cumlHandle& handle,
                      RandomForestMetaData[double, double]*,
                      double*,
                      int,
                      int,
                      double*,
                      int) except +

    cdef RF_metrics score(cumlHandle& handle,
                          RandomForestMetaData[float, float]*,
                          float*,
                          int,
                          float*,
                          int) except +

    cdef RF_metrics score(cumlHandle& handle,
                          RandomForestMetaData[double, double]*,
                          double*,
                          int,
                          double*,
                          int) except +


class RandomForestRegressor(BaseRandomForestModel, RegressorMixin):

    """
    Implements a Random Forest regressor model which fits multiple decision
    trees in an ensemble.
    Note that the underlying algorithm for tree node splits differs from that
    used in scikit-learn. By default, the cuML Random Forest uses a
    histogram-based algorithm to determine splits, rather than an exact
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
    n_estimators : int (default = 100)
        Number of trees in the forest. (Default changed to 100 in cuML 0.11)
    handle : cuml.Handle
        If it is None, a new one is created just for this class.
    split_algo : int (default = 1)
        The algorithm to determine how nodes are split in the tree.
        0 for HIST and 1 for GLOBAL_QUANTILE. HIST curently uses a slower
        tree-building algorithm so GLOBAL_QUANTILE is recommended for most
        cases.
    split_criterion : int (default = 2)
        The criterion used to split nodes.
        0 for GINI, 1 for ENTROPY,
        2 for MSE, or 3 for MAE
        0 and 1 not valid for regression
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
        Ratio of number of features (columns) to consider
        per node split.
        If int then max_features/n_features.
        If float then max_features is used as a fraction.
        If 'auto' then max_features=1.0.
        If 'sqrt' then max_features=1/sqrt(n_features).
        If 'log2' then max_features=log2(n_features)/n_features.
    n_bins : int (default = 8)
        Number of bins used by the split algorithm.
    min_rows_per_node : int or float (default = 2)
        The minimum number of samples (rows) needed to split a node.
        If int then number of sample rows
        If float the min_rows_per_sample*n_rows
    min_impurity_decrease : float (default = 0.0)
        The minimum decrease in impurity required for node to be split
    accuracy_metric : string (default = 'mse')
        Decides the metric used to evaluate the performance of the model.
        for median of abs error : 'median_ae'
        for mean of abs error : 'mean_ae'
        for mean square error' : 'mse'
    quantile_per_tree : boolean (default = False)
        Whether quantile is computed for individal trees in RF.
        Only relevant for GLOBAL_QUANTILE split_algo.
    seed : int (default = None)
        Seed for the random number generator. Unseeded by default. Does not
        currently fully guarantee the exact same results.

    """

    def __init__(self, split_criterion=2,
                 accuracy_metric='mse',
                 **kwargs):
        self.RF_type = REGRESSION
        super(RandomForestRegressor, self).__init__(
            split_criterion=split_criterion,
            accuracy_metric=accuracy_metric,
            **kwargs)
    """
    TODO:
        Add the preprocess and postprocess functions
        in the cython code to normalize the labels
    """
    def __getstate__(self):
        state = self.__dict__.copy()
        cdef size_t params_t
        cdef  RandomForestMetaData[float, float] *rf_forest
        cdef  RandomForestMetaData[double, double] *rf_forest64
        cdef size_t params_t64
        if self.n_cols:
            # only if model has been fit previously
            self._get_serialized_model()  # Ensure we have this cached
            if self.rf_forest:
                params_t = <uintptr_t> self.rf_forest
                rf_forest = \
                    <RandomForestMetaData[float, float]*>params_t
                state["rf_params"] = rf_forest.rf_params

            if self.rf_forest64:
                params_t64 = <uintptr_t> self.rf_forest64
                rf_forest64 = \
                    <RandomForestMetaData[double, double]*>params_t64
                state["rf_params64"] = rf_forest64.rf_params

        state['n_cols'] = self.n_cols
        state["verbose"] = self.verbose
        state["treelite_serialized_model"] = self.treelite_serialized_model
        state['handle'] = self.handle
        state["treelite_handle"] = None
        state["split_criterion"] = self.split_criterion

        return state

    def __setstate__(self, state):
        super(RandomForestRegressor, self).__init__(
            split_criterion=state["split_criterion"],
            handle=state["handle"], verbose=state['verbose'])
        cdef  RandomForestMetaData[float, float] *rf_forest = \
            new RandomForestMetaData[float, float]()
        cdef  RandomForestMetaData[double, double] *rf_forest64 = \
            new RandomForestMetaData[double, double]()

        self.n_cols = state['n_cols']
        if self.n_cols:
            rf_forest.rf_params = state["rf_params"]
            state["rf_forest"] = <uintptr_t>rf_forest

            rf_forest64.rf_params = state["rf_params64"]
            state["rf_forest64"] = <uintptr_t>rf_forest64

        self.treelite_serialized_model = state["treelite_serialized_model"]
        self.__dict__.update(state)

    def __del__(self):
        self._reset_forest_data()

    def _reset_forest_data(self):
        """Free memory allocated by this instance and clear instance vars."""
        if self.rf_forest:
            delete_rf_metadata(
                <RandomForestMetaData[float, float]*><uintptr_t>
                self.rf_forest)
            self.rf_forest = 0
        if self.rf_forest64:
            delete_rf_metadata(
                <RandomForestMetaData[double, double]*><uintptr_t>
                self.rf_forest64)
            self.rf_forest64 = 0

        if self.treelite_handle:
            TreeliteModel.free_treelite_model(self.treelite_handle)

        self.treelite_handle = None
        self.treelite_serialized_model = None
        self.n_cols = None

    def convert_to_treelite_model(self):
        """
        Converts the cuML RF model to a Treelite model

        Returns
        ----------
        tl_to_fil_model : Treelite version of this model
        """
        treelite_handle = self._obtain_treelite_handle()
        return TreeliteModel.from_treelite_model_handle(treelite_handle)

    def convert_to_fil_model(self, output_class=False,
                             algo='auto',
                             fil_sparse_format='auto'):
        """
        Create a Forest Inference (FIL) model from the trained cuML
        Random Forest model.
        Parameters
        ----------
        output_class : boolean (default = False)
            This is optional and required only while performing the
            predict operation on the GPU.
            If true, return a 1 or 0 depending on whether the raw
            prediction exceeds the threshold. If False, just return
            the raw prediction.
        algo : string (default = 'auto')
            This is optional and required only while performing the
            predict operation on the GPU.
            'naive' - simple inference using shared memory
            'tree_reorg' - similar to naive but trees rearranged to be more
            coalescing-friendly
            'batch_tree_reorg' - similar to tree_reorg but predicting
            multiple rows per thread block
            `auto` - choose the algorithm automatically. Currently
            'batch_tree_reorg' is used for dense storage
            and 'naive' for sparse storage
        fil_sparse_format : boolean or string (default = 'auto')
            This variable is used to choose the type of forest that will be
            created in the Forest Inference Library. It is not required
            while using predict_model='CPU'.
            'auto' - choose the storage type automatically
            (currently True is chosen by auto)
            False - create a dense forest
            True - create a sparse forest, requires algo='naive'
            or algo='auto'
        Returns
        ----------
        fil_model :
            A Forest Inference model which can be used to perform
            inferencing on the random forest model.
        """
        treelite_handle = self._obtain_treelite_handle()
        return _obtain_fil_model(treelite_handle=treelite_handle,
                                 depth=self.max_depth,
                                 output_class=output_class,
                                 algo=algo,
                                 fil_sparse_format=fil_sparse_format)

    """
    TODO : Move functions duplicated in the RF classifier and regressor
           to a shared file. Cuml issue #1854 has been created to track this.
    """

    def fit(self, X, y, convert_dtype=True):
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
        convert_dtype : bool, optional (default = True)
            When set to True, the fit method will, when necessary, convert
            y to be the same data type as X if they differ. This will increase
            memory used for the method.
        """
        X_m, y_m, max_feature_val = self._dataset_setup_for_fit(X, y,
                                                                convert_dtype)

        # Reset the old tree data for new fit call
        cdef uintptr_t X_ptr, y_ptr
        X_ptr = X_m.ptr
        y_ptr = y_m.ptr

        cdef cumlHandle* handle_ =\
            <cumlHandle*><uintptr_t>self.handle.getHandle()

        cdef RandomForestMetaData[float, float] *rf_forest = \
            new RandomForestMetaData[float, float]()
        self.rf_forest = <uintptr_t> rf_forest
        cdef RandomForestMetaData[double, double] *rf_forest64 = \
            new RandomForestMetaData[double, double]()
        self.rf_forest64 = <uintptr_t> rf_forest64
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
                <int> self.n_rows,
                <int> self.n_cols,
                <float*> y_ptr,
                rf_params,
                <int> self.verbose)

        else:
            rf_params64 = rf_params
            fit(handle_[0],
                rf_forest64,
                <double*> X_ptr,
                <int> self.n_rows,
                <int> self.n_cols,
                <double*> y_ptr,
                rf_params64,
                <int> self.verbose)
        # make sure that the `fit` is complete before the following delete
        # call happens
        self.handle.sync()
        del X_m
        del y_m
        return self

    def _predict_model_on_cpu(self, X, convert_dtype):
        out_type = self._get_output_type(X)
        cdef uintptr_t X_ptr
        X_m, n_rows, n_cols, dtype = \
            input_to_cuml_array(X, order='C',
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None),
                                check_cols=self.n_cols)
        X_ptr = X_m.ptr

        preds = CumlArray.zeros(n_rows, dtype=dtype)
        cdef uintptr_t preds_ptr = preds.ptr

        cdef cumlHandle* handle_ =\
            <cumlHandle*><uintptr_t>self.handle.getHandle()

        cdef RandomForestMetaData[float, float] *rf_forest = \
            <RandomForestMetaData[float, float]*><uintptr_t> self.rf_forest

        cdef RandomForestMetaData[double, double] *rf_forest64 = \
            <RandomForestMetaData[double, double]*><uintptr_t> self.rf_forest64
        if self.dtype == np.float32:
            predict(handle_[0],
                    rf_forest,
                    <float*> X_ptr,
                    <int> n_rows,
                    <int> n_cols,
                    <float*> preds_ptr,
                    <int> self.verbose)

        elif self.dtype == np.float64:
            predict(handle_[0],
                    rf_forest64,
                    <double*> X_ptr,
                    <int> n_rows,
                    <int> n_cols,
                    <double*> preds_ptr,
                    <int> self.verbose)
        else:
            raise TypeError("supports only float32 and float64 input,"
                            " but input of type '%s' passed."
                            % (str(self.dtype)))

        self.handle.sync()
        # synchronous w/o a stream
        del(X_m)
        return preds.to_output(out_type)

    def predict(self, X, predict_model="GPU",
                algo='auto', convert_dtype=True,
                fil_sparse_format='auto'):
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
        algo : string (default = 'auto')
            This is optional and required only while performing the
            predict operation on the GPU.
            'naive' - simple inference using shared memory
            'tree_reorg' - similar to naive but trees rearranged to be more
            coalescing-friendly
            'batch_tree_reorg' - similar to tree_reorg but predicting
            multiple rows per thread block
            `auto` - choose the algorithm automatically. Currently
            'batch_tree_reorg' is used for dense storage
            and 'naive' for sparse storage
        convert_dtype : bool, optional (default = True)
            When set to True, the predict method will, when necessary, convert
            the input to the data type which was used to train the model. This
            will increase memory used for the method.
        fil_sparse_format : boolean or string (default = auto)
            This variable is used to choose the type of forest that will be
            created in the Forest Inference Library. It is not required
            while using predict_model='CPU'.
            'auto' - choose the storage type automatically
            (currently True is chosen by auto)
            False - create a dense forest
            True - create a sparse forest, requires algo='naive'
            or algo='auto'

        Returns
        ----------
        y : NumPy
            Dense vector (int) of shape (n_samples, 1)

        """
        if predict_model == "CPU":
            preds = self._predict_model_on_cpu(X, convert_dtype)

        elif self.dtype == np.float64:
            raise TypeError("GPU based predict only accepts np.float32 data. \
                            In order use the GPU predict the model should \
                            also be trained using a np.float32 dataset. \
                            If you would like to use np.float64 dtype \
                            then please use the CPU based predict by \
                            setting predict_model = 'CPU'")

        else:
            preds = self._predict_model_on_gpu(
                X=X,
                algo=algo,
                convert_dtype=convert_dtype,
                fil_sparse_format=fil_sparse_format)

        return preds

    def score(self, X, y, algo='auto', convert_dtype=True,
              fil_sparse_format='auto', predict_model="GPU"):
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
        algo : string (default = 'auto')
            This is optional and required only while performing the
            predict operation on the GPU.
            'naive' - simple inference using shared memory
            'tree_reorg' - similar to naive but trees rearranged to be more
            coalescing-friendly
            'batch_tree_reorg' - similar to tree_reorg but predicting
            multiple rows per thread block
            `auto` - choose the algorithm automatically. Currently
            'batch_tree_reorg' is used for dense storage
            and 'naive' for sparse storage
        convert_dtype : boolean, default=True
            whether to convert input data to correct dtype automatically
        predict_model : String (default = 'GPU')
            'GPU' to predict using the GPU, 'CPU' otherwise. The GPU can only
            be used if the model was trained on float32 data and `X` is float32
            or convert_dtype is set to True.
        fil_sparse_format : boolean or string (default = auto)
            This variable is used to choose the type of forest that will be
            created in the Forest Inference Library. It is not required
            while using predict_model='CPU'.
            'auto' - choose the storage type automatically
            (currently True is chosen by auto)
            False - create a dense forest
            True - create a sparse forest, requires algo='naive'
            or algo='auto'

        Returns
        ----------
        mean_square_error : float or
        median_abs_error : float or
        mean_abs_error : float
        """
        cdef uintptr_t y_ptr
        _, n_rows, _, dtype = \
            input_to_cuml_array(X,
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None))
        y_m, n_rows, _, y_dtype = \
            input_to_cuml_array(y,
                                convert_to_dtype=(dtype if convert_dtype
                                                  else False))
        y_ptr = y_m.ptr
        preds = self.predict(X, algo=algo,
                             convert_dtype=convert_dtype,
                             fil_sparse_format=fil_sparse_format,
                             predict_model=predict_model)

        cdef uintptr_t preds_ptr
        preds_m, _, _, _ = \
            input_to_cuml_array(preds, convert_to_dtype=dtype)
        preds_ptr = preds_m.ptr

        cdef cumlHandle* handle_ =\
            <cumlHandle*><uintptr_t>self.handle.getHandle()

        cdef RandomForestMetaData[float, float] *rf_forest = \
            <RandomForestMetaData[float, float]*><uintptr_t> self.rf_forest

        cdef RandomForestMetaData[double, double] *rf_forest64 = \
            <RandomForestMetaData[double, double]*><uintptr_t> self.rf_forest64

        if self.dtype == np.float32:
            self.temp_stats = score(handle_[0],
                                    rf_forest,
                                    <float*> y_ptr,
                                    <int> n_rows,
                                    <float*> preds_ptr,
                                    <int> self.verbose)

        elif self.dtype == np.float64:
            self.temp_stats = score(handle_[0],
                                    rf_forest64,
                                    <double*> y_ptr,
                                    <int> n_rows,
                                    <double*> preds_ptr,
                                    <int> self.verbose)

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
        return self._get_params(deep=deep)

    def set_params(self, **params):
        """
        Sets the value of parameters required to
        configure this estimator, it functions similar to
        the sklearn set_params.
        Parameters
        -----------
        params : dict of new params
        """
        return self._set_params(**params)

    def print_summary(self):
        """
        Prints the summary of the forest used to train and test the model
        """
        cdef RandomForestMetaData[float, float] *rf_forest = \
            <RandomForestMetaData[float, float]*><uintptr_t> self.rf_forest

        cdef RandomForestMetaData[double, double] *rf_forest64 = \
            <RandomForestMetaData[double, double]*><uintptr_t> self.rf_forest64

        if self.dtype == np.float64:
            print_rf_summary(rf_forest64)
        else:
            print_rf_summary(rf_forest)

    def print_detailed(self):
        """
        Prints the detailed information about the forest used to
        train and test the Random Forest model
        """
        cdef RandomForestMetaData[float, float] *rf_forest = \
            <RandomForestMetaData[float, float]*><uintptr_t> self.rf_forest

        cdef RandomForestMetaData[double, double] *rf_forest64 = \
            <RandomForestMetaData[double, double]*><uintptr_t> self.rf_forest64

        if self.dtype == np.float64:
            print_rf_detailed(rf_forest64)
        else:
            print_rf_detailed(rf_forest)

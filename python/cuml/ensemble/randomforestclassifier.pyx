
#
# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

# distutils: language = c++

import numpy as np
import rmm
import warnings

import cuml.common.logger as logger

from cuml import ForestInference
from cuml.common.array import CumlArray
from cuml.common.mixins import ClassifierMixin
import cuml.internals
from cuml.common.doc_utils import generate_docstring
from cuml.common.doc_utils import insert_into_docstring
from cuml.raft.common.handle import Handle
from cuml.common import input_to_cuml_array

from cuml.ensemble.randomforest_common import BaseRandomForestModel
from cuml.ensemble.randomforest_common import _obtain_fil_model
from cuml.ensemble.randomforest_shared cimport *

from cuml.fil.fil import TreeliteModel

from cython.operator cimport dereference as deref

from libcpp cimport bool
from libcpp.vector cimport vector
from libc.stdint cimport uintptr_t, uint64_t
from libc.stdlib cimport calloc, malloc, free

from numba import cuda

from cuml.common.cuda import nvtx_range_wrap, nvtx_range_push, nvtx_range_pop
from cuml.raft.common.handle cimport handle_t
cimport cuml.common.cuda

cimport cython


cdef extern from "cuml/ensemble/randomforest.hpp" namespace "ML":

    cdef void fit(handle_t& handle,
                  RandomForestMetaData[float, int]*,
                  float*,
                  int,
                  int,
                  int*,
                  int,
                  RF_params,
                  int) except +

    cdef void fit(handle_t& handle,
                  RandomForestMetaData[double, int]*,
                  double*,
                  int,
                  int,
                  int*,
                  int,
                  RF_params,
                  int) except +

    cdef void predict(handle_t& handle,
                      RandomForestMetaData[float, int] *,
                      float*,
                      int,
                      int,
                      int*,
                      bool) except +

    cdef void predict(handle_t& handle,
                      RandomForestMetaData[double, int]*,
                      double*,
                      int,
                      int,
                      int*,
                      bool) except +

    cdef void predictGetAll(handle_t& handle,
                            RandomForestMetaData[float, int] *,
                            float*,
                            int,
                            int,
                            int*,
                            bool) except +

    cdef void predictGetAll(handle_t& handle,
                            RandomForestMetaData[double, int]*,
                            double*,
                            int,
                            int,
                            int*,
                            bool) except +

    cdef RF_metrics score(handle_t& handle,
                          RandomForestMetaData[float, int]*,
                          int*,
                          int,
                          int*,
                          bool) except +

    cdef RF_metrics score(handle_t& handle,
                          RandomForestMetaData[double, int]*,
                          int*,
                          int,
                          int*,
                          bool) except +


class RandomForestClassifier(BaseRandomForestModel,
                             ClassifierMixin):
    """
    Implements a Random Forest classifier model which fits multiple decision
    tree classifiers in an ensemble.

    .. note:: Note that the underlying algorithm for tree node splits differs
      from that used in scikit-learn. By default, the cuML Random Forest uses a
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
      * While training the model for multi class classification problems,
        using deep trees or `max_features=1.0` provides better performance.

    Examples
    --------
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
    split_criterion : The criterion used to split nodes.
        0 for GINI, 1 for ENTROPY
        2 and 3 not valid for classification
        (default = 0)
    split_algo : int (default = 1)
        The algorithm to determine how nodes are split in the tree.
        0 for HIST and 1 for GLOBAL_QUANTILE. HIST currently uses a slower
        tree-building algorithm so GLOBAL_QUANTILE is recommended for most
        cases.
    bootstrap : boolean (default = True)
        Control bootstrapping.
        If True, each tree in the forest is built
        on a bootstrapped sample with replacement.
        If False, the whole dataset is used to build each tree.
    bootstrap_features : boolean (default = False)
        Control bootstrapping for features.
        If features are drawn with or without replacement
    max_samples : float (default = 1.0)
        Ratio of dataset rows used while fitting each tree.
    max_depth : int (default = 16)
        Maximum tree depth. Unlimited (i.e, until leaves are pure),
        if -1. Unlimited depth is not supported.
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
    n_bins : int (default = 8)
        Number of bins used by the split algorithm.
    min_samples_leaf : int or float (default = 1)
        The minimum number of samples (rows) in each leaf node.
        If int, then min_samples_leaf represents the minimum number.
        If float, then min_samples_leaf represents a fraction and
        ceil(min_samples_leaf * n_rows) is the minimum number of samples
        for each leaf node.
    min_samples_split : int or float (default = 2)
        The minimum number of samples required to split an internal node.
        If int, then min_samples_split represents the minimum number.
        If float, then min_samples_split represents a fraction and
        ceil(min_samples_split * n_rows) is the minimum number of samples
        for each split.
    min_impurity_decrease : float (default = 0.0)
        Minimum decrease in impurity requried for
        node to be spilt.
    quantile_per_tree : boolean (default = False)
        Whether quantile is computed for individual trees in RF.
        Only relevant when `split_algo = GLOBAL_QUANTILE`.

        .. deprecated:: 0.19
           Parameter 'quantile_per_tree' is deprecated and will be removed in
           subsequent release.
    use_experimental_backend : boolean (default = False)
        If set to true and the following conditions are also met, experimental
        decision tree training implementation would be used only if
        `split_algo = 1` (GLOBAL_QUANTILE) and `quantile_per_tree = False`
        (No per tree quantile computation).
    max_batch_size: int (default = 128)
        Maximum number of nodes that can be processed in a given batch. This is
        used only when 'use_experimental_backend' is true. Does not currently
        fully guarantee the exact same results.
    random_state : int (default = None)
        Seed for the random number generator. Unseeded by default. Does not
        currently fully guarantee the exact same results. **Note: Parameter
        `seed` is removed since release 0.19.**

    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {'input', 'cudf', 'cupy', 'numpy', 'numba'}, default=None
        Variable to control output type of the results and attributes of
        the estimator. If None, it'll inherit the output type set at the
        module level, `cuml.global_settings.output_type`.
        See :ref:`output-data-type-configuration` for more info.

    """

    def __init__(self, *, split_criterion=0, handle=None, verbose=False,
                 output_type=None, **kwargs):

        self.RF_type = CLASSIFICATION
        self.num_classes = 2
        super().__init__(
            split_criterion=split_criterion,
            handle=handle,
            verbose=verbose,
            output_type=output_type,
            **kwargs)

    """
    TODO:
        Add the preprocess and postprocess functions
        in the cython code to normalize the labels
        Link to the above issue on github :
        https://github.com/rapidsai/cuml/issues/691
    """
    def __getstate__(self):
        state = self.__dict__.copy()
        cdef size_t params_t
        cdef  RandomForestMetaData[float, int] *rf_forest
        cdef  RandomForestMetaData[double, int] *rf_forest64
        cdef size_t params_t64
        if self.n_cols:
            # only if model has been fit previously
            self._get_serialized_model()  # Ensure we have this cached
            if self.rf_forest:
                params_t = <uintptr_t> self.rf_forest
                rf_forest = \
                    <RandomForestMetaData[float, int]*>params_t
                state["rf_params"] = rf_forest.rf_params

            if self.rf_forest64:
                params_t64 = <uintptr_t> self.rf_forest64
                rf_forest64 = \
                    <RandomForestMetaData[double, int]*>params_t64
                state["rf_params64"] = rf_forest64.rf_params

        state["n_cols"] = self.n_cols
        state["verbose"] = self.verbose
        state["treelite_serialized_model"] = self.treelite_serialized_model
        state["treelite_handle"] = None
        state["split_criterion"] = self.split_criterion
        state["handle"] = self.handle
        return state

    def __setstate__(self, state):
        super(RandomForestClassifier, self).__init__(
            split_criterion=state["split_criterion"],
            handle=state["handle"],
            verbose=state["verbose"])
        cdef  RandomForestMetaData[float, int] *rf_forest = \
            new RandomForestMetaData[float, int]()
        cdef  RandomForestMetaData[double, int] *rf_forest64 = \
            new RandomForestMetaData[double, int]()

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
                <RandomForestMetaData[float, int]*><uintptr_t>
                self.rf_forest)
            self.rf_forest = 0
        if self.rf_forest64:
            delete_rf_metadata(
                <RandomForestMetaData[double, int]*><uintptr_t>
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

    def convert_to_fil_model(self, output_class=True,
                             threshold=0.5, algo='auto',
                             fil_sparse_format='auto'):
        """
        Create a Forest Inference (FIL) model from the trained cuML
        Random Forest model.

        Parameters
        ----------

        output_class : boolean (default = True)
            This is optional and required only while performing the
            predict operation on the GPU.
            If true, return a 1 or 0 depending on whether the raw
            prediction exceeds the threshold. If False, just return
            the raw prediction.
        algo : string (default = 'auto')
            This is optional and required only while performing the
            predict operation on the GPU.

             * ``'naive'`` - simple inference using shared memory
             * ``'tree_reorg'`` - similar to naive but trees rearranged to be
               more coalescing-friendly
             * ``'batch_tree_reorg'`` - similar to tree_reorg but predicting
               multiple rows per thread block
             * ``'auto'`` - choose the algorithm automatically. Currently
             * ``'batch_tree_reorg'`` is used for dense storage
               and 'naive' for sparse storage

        threshold : float (default = 0.5)
            Threshold used for classification. Optional and required only
            while performing the predict operation on the GPU.
            It is applied if output_class == True, else it is ignored
        fil_sparse_format : boolean or string (default = auto)
            This variable is used to choose the type of forest that will be
            created in the Forest Inference Library. It is not required
            while using predict_model='CPU'.

             * ``'auto'`` - choose the storage type automatically
               (currently True is chosen by auto)
             * ``False`` - create a dense forest
             * ``True`` - create a sparse forest, requires algo='naive'
               or algo='auto'

        Returns
        -------

        fil_model
            A Forest Inference model which can be used to perform
            inferencing on the random forest model.

        """
        treelite_handle = self._obtain_treelite_handle()
        return _obtain_fil_model(treelite_handle=treelite_handle,
                                 depth=self.max_depth,
                                 output_class=output_class,
                                 threshold=threshold,
                                 algo=algo,
                                 fil_sparse_format=fil_sparse_format)

    @generate_docstring(skip_parameters_heading=True,
                        y='dense_intdtype',
                        convert_dtype_cast='np.float32')
    @cuml.internals.api_base_return_any(set_output_type=False,
                                        set_output_dtype=True,
                                        set_n_features_in=False)
    def fit(self, X, y, convert_dtype=True):
        """
        Perform Random Forest Classification on the input data

        Parameters
        ----------
        convert_dtype : bool, optional (default = True)
            When set to True, the fit method will, when necessary, convert
            y to be of dtype int32. This will increase memory used for
            the method.
        """
        nvtx_range_push("Fit RF-Classifier @randomforestclassifier.pyx")

        X_m, y_m, max_feature_val = self._dataset_setup_for_fit(X, y,
                                                                convert_dtype)
        cdef uintptr_t X_ptr, y_ptr

        X_ptr = X_m.ptr
        y_ptr = y_m.ptr

        cdef handle_t* handle_ =\
            <handle_t*><uintptr_t>self.handle.getHandle()

        cdef RandomForestMetaData[float, int] *rf_forest = \
            new RandomForestMetaData[float, int]()
        self.rf_forest = <uintptr_t> rf_forest
        cdef RandomForestMetaData[double, int] *rf_forest64 = \
            new RandomForestMetaData[double, int]()
        self.rf_forest64 = <uintptr_t> rf_forest64

        if self.random_state is None:
            seed_val = <uintptr_t>NULL
        else:
            seed_val = <uintptr_t>self.random_state

        rf_params = set_rf_params(<int> self.max_depth,
                                  <int> self.max_leaves,
                                  <float> max_feature_val,
                                  <int> self.n_bins,
                                  <int> self.split_algo,
                                  <int> self.min_samples_leaf,
                                  <int> self.min_samples_split,
                                  <float> self.min_impurity_decrease,
                                  <bool> self.bootstrap_features,
                                  <bool> self.bootstrap,
                                  <int> self.n_estimators,
                                  <float> self.max_samples,
                                  <uint64_t> seed_val,
                                  <CRITERION> self.split_criterion,
                                  <bool> self.quantile_per_tree,
                                  <int> self.n_streams,
                                  <bool> self.use_experimental_backend,
                                  <int> self.max_batch_size)

        if self.dtype == np.float32:
            fit(handle_[0],
                rf_forest,
                <float*> X_ptr,
                <int> self.n_rows,
                <int> self.n_cols,
                <int*> y_ptr,
                <int> self.num_classes,
                rf_params,
                <int> self.verbose)

        elif self.dtype == np.float64:
            rf_params64 = rf_params
            fit(handle_[0],
                rf_forest64,
                <double*> X_ptr,
                <int> self.n_rows,
                <int> self.n_cols,
                <int*> y_ptr,
                <int> self.num_classes,
                rf_params64,
                <int> self.verbose)

        else:
            raise TypeError("supports only np.float32 and np.float64 input,"
                            " but input of type '%s' passed."
                            % (str(self.dtype)))
        # make sure that the `fit` is complete before the following delete
        # call happens
        self.handle.sync()
        del X_m
        del y_m
        nvtx_range_pop()
        return self

    @cuml.internals.api_base_return_array(get_output_dtype=True)
    def _predict_model_on_cpu(self, X, convert_dtype) -> CumlArray:
        cdef uintptr_t X_ptr
        X_m, n_rows, n_cols, dtype = \
            input_to_cuml_array(X, order='C',
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None),
                                check_cols=self.n_cols)
        X_ptr = X_m.ptr
        preds = CumlArray.zeros(n_rows, dtype=np.int32)
        cdef uintptr_t preds_ptr = preds.ptr

        cdef handle_t* handle_ =\
            <handle_t*><uintptr_t>self.handle.getHandle()

        cdef RandomForestMetaData[float, int] *rf_forest = \
            <RandomForestMetaData[float, int]*><uintptr_t> self.rf_forest

        cdef RandomForestMetaData[double, int] *rf_forest64 = \
            <RandomForestMetaData[double, int]*><uintptr_t> self.rf_forest64
        if self.dtype == np.float32:
            predict(handle_[0],
                    rf_forest,
                    <float*> X_ptr,
                    <int> n_rows,
                    <int> n_cols,
                    <int*> preds_ptr,
                    <int> self.verbose)

        elif self.dtype == np.float64:
            predict(handle_[0],
                    rf_forest64,
                    <double*> X_ptr,
                    <int> n_rows,
                    <int> n_cols,
                    <int*> preds_ptr,
                    <int> self.verbose)
        else:
            raise TypeError("supports only np.float32 and np.float64 input,"
                            " but input of type '%s' passed."
                            % (str(self.dtype)))

        self.handle.sync()
        # synchronous w/o a stream
        del(X_m)
        return preds

    @insert_into_docstring(parameters=[('dense', '(n_samples, n_features)')],
                           return_values=[('dense', '(n_samples, 1)')])
    def predict(self, X, predict_model="GPU", threshold=0.5,
                algo='auto', num_classes=None,
                convert_dtype=True,
                fil_sparse_format='auto') -> CumlArray:
        """
        Predicts the labels for X.

        Parameters
        ----------
        X : {}
        predict_model : String (default = 'GPU')
            'GPU' to predict using the GPU, 'CPU' otherwise. The 'GPU' can only
            be used if the model was trained on float32 data and `X` is float32
            or convert_dtype is set to True. Also the 'GPU' should only be
            used for classification problems.
        algo : string (default = ``'auto'``)
            This is optional and required only while performing the
            predict operation on the GPU.

             * ``'naive'`` - simple inference using shared memory
             * ``'tree_reorg'`` - similar to naive but trees rearranged to be
               more coalescing-friendly
             * ``'batch_tree_reorg'`` - similar to tree_reorg but predicting
               multiple rows per thread block
             * ``'auto'`` - choose the algorithm automatically. Currently
             * ``'batch_tree_reorg'`` is used for dense storage
               and 'naive' for sparse storage

        threshold : float (default = 0.5)
            Threshold used for classification. Optional and required only
            while performing the predict operation on the GPU.
        num_classes : int (default = None)
            number of different classes present in the dataset.

            .. deprecated:: 0.16
                Parameter 'num_classes' is deprecated and will be removed in
                an upcoming version. The number of classes passed must match
                the number of classes the model was trained on.

        convert_dtype : bool, optional (default = True)
            When set to True, the predict method will, when necessary, convert
            the input to the data type which was used to train the model. This
            will increase memory used for the method.
        fil_sparse_format : boolean or string (default = ``'auto'``)
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
        y : {}
        """
        nvtx_range_push("predict RF-Classifier @randomforestclassifier.pyx")
        if num_classes:
            warnings.warn("num_classes is deprecated and will be removed"
                          " in an upcoming version")
            if num_classes != self.num_classes:
                raise NotImplementedError("limiting num_classes for predict"
                                          " is not implemented")
        if predict_model == "CPU":
            preds = self._predict_model_on_cpu(X,
                                               convert_dtype=convert_dtype)

        elif self.dtype == np.float64:
            raise TypeError("GPU based predict only accepts np.float32 data. \
                            In order use the GPU predict the model should \
                            also be trained using a np.float32 dataset. \
                            If you would like to use np.float64 dtype \
                            then please use the CPU based predict by \
                            setting predict_model = 'CPU'")

        else:
            preds = \
                self._predict_model_on_gpu(X=X, output_class=True,
                                           threshold=threshold,
                                           algo=algo,
                                           convert_dtype=convert_dtype,
                                           fil_sparse_format=fil_sparse_format,
                                           predict_proba=False)

        nvtx_range_pop()
        return preds

    def _predict_get_all(self, X, convert_dtype=True) -> CumlArray:
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
        X_m, n_rows, n_cols, dtype = \
            input_to_cuml_array(X, order='C',
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None),
                                check_cols=self.n_cols)
        X_ptr = X_m.ptr

        preds = CumlArray.zeros(n_rows * self.n_estimators, dtype=np.int32)
        preds_ptr = preds.ptr

        cdef handle_t* handle_ =\
            <handle_t*><uintptr_t>self.handle.getHandle()
        cdef RandomForestMetaData[float, int] *rf_forest = \
            <RandomForestMetaData[float, int]*><uintptr_t> self.rf_forest

        cdef RandomForestMetaData[double, int] *rf_forest64 = \
            <RandomForestMetaData[double, int]*><uintptr_t> self.rf_forest64
        if self.dtype == np.float32:
            predictGetAll(handle_[0],
                          rf_forest,
                          <float*> X_ptr,
                          <int> n_rows,
                          <int> n_cols,
                          <int*> preds_ptr,
                          <int> self.verbose)

        elif self.dtype == np.float64:
            predictGetAll(handle_[0],
                          rf_forest64,
                          <double*> X_ptr,
                          <int> n_rows,
                          <int> n_cols,
                          <int*> preds_ptr,
                          <int> self.verbose)
        else:
            raise TypeError("supports only np.float32 and np.float64 input,"
                            " but input of type '%s' passed."
                            % (str(self.dtype)))
        self.handle.sync()
        del(X_m)
        return preds

    @insert_into_docstring(parameters=[('dense', '(n_samples, n_features)')],
                           return_values=[('dense', '(n_samples, 1)')])
    def predict_proba(self, X, algo='auto',
                      num_classes=None, convert_dtype=True,
                      fil_sparse_format='auto') -> CumlArray:
        """
        Predicts class probabilites for X. This function uses the GPU
        implementation of predict. Therefore, data with 'dtype = np.float32'
        should be used with this function.

        Parameters
        ----------
        X : {}
        algo : string (default = 'auto')
            This is optional and required only while performing the
            predict operation on the GPU.

             * ``'naive'`` - simple inference using shared memory
             * ``'tree_reorg'`` - similar to naive but trees rearranged to be
               more coalescing-friendly
             * ``'batch_tree_reorg'`` - similar to tree_reorg but predicting
               multiple rows per thread block
             * ``'auto'`` - choose the algorithm automatically. Currently
             * ``'batch_tree_reorg'`` is used for dense storage
               and 'naive' for sparse storage

        num_classes : int (default = None)
            number of different classes present in the dataset.

            .. deprecated:: 0.16
                Parameter 'num_classes' is deprecated and will be removed in
                an upcoming version. The number of classes passed must match
                the number of classes the model was trained on.

        convert_dtype : bool, optional (default = True)
            When set to True, the predict method will, when necessary, convert
            the input to the data type which was used to train the model. This
            will increase memory used for the method.
        fil_sparse_format : boolean or string (default = auto)
            This variable is used to choose the type of forest that will be
            created in the Forest Inference Library. It is not required
            while using predict_model='CPU'.

             * ``'auto'`` - choose the storage type automatically
               (currently True is chosen by auto)
             * ``False`` - create a dense forest
             * ``True`` - create a sparse forest, requires algo='naive'
               or algo='auto'

        Returns
        -------
        y : {}
        """
        if self.dtype == np.float64:
            raise TypeError("GPU based predict only accepts np.float32 data. \
                            In order use the GPU predict the model should \
                            also be trained using a np.float32 dataset. \
                            If you would like to use np.float64 dtype \
                            then please use the CPU based predict by \
                            setting predict_model = 'CPU'")

        if num_classes:
            warnings.warn("num_classes is deprecated and will be removed"
                          " in an upcoming version")
            if num_classes != self.num_classes:
                raise NotImplementedError("The number of classes in the test "
                                          "dataset should be equal to the "
                                          "number of classes present in the "
                                          "training dataset.")

        preds_proba = \
            self._predict_model_on_gpu(X, output_class=True,
                                       algo=algo,
                                       convert_dtype=convert_dtype,
                                       fil_sparse_format=fil_sparse_format,
                                       predict_proba=True)

        return preds_proba

    @insert_into_docstring(parameters=[('dense', '(n_samples, n_features)'),
                                       ('dense_intdtype', '(n_samples, 1)')])
    def score(self, X, y, threshold=0.5,
              algo='auto', num_classes=None, predict_model="GPU",
              convert_dtype=True, fil_sparse_format='auto'):
        """
        Calculates the accuracy metric score of the model for X.

        Parameters
        ----------
        X : {}
        y : {}
        algo : string (default = 'auto')
            This is optional and required only while performing the
            predict operation on the GPU.

             * ``'naive'`` - simple inference using shared memory
             * ``'tree_reorg'`` - similar to naive but trees rearranged to be
               more coalescing-friendly
             * ``'batch_tree_reorg'`` - similar to tree_reorg but predicting
               multiple rows per thread block
             * ``'auto'`` - choose the algorithm automatically. Currently
             * ``'batch_tree_reorg'`` is used for dense storage
               and 'naive' for sparse storage

        threshold : float
            threshold is used to for classification
            This is optional and required only while performing the
            predict operation on the GPU.
        num_classes : int (default = None)
            number of different classes present in the dataset.

            .. deprecated:: 0.16
                Parameter 'num_classes' is deprecated and will be removed in
                an upcoming version. The number of classes passed must match
                the number of classes the model was trained on.

        convert_dtype : boolean, default=True
            whether to convert input data to correct dtype automatically
        predict_model : String (default = 'GPU')
            'GPU' to predict using the GPU, 'CPU' otherwise. The 'GPU' can only
            be used if the model was trained on float32 data and `X` is float32
            or convert_dtype is set to True. Also the 'GPU' should only be
            used for classification problems.
        fil_sparse_format : boolean or string (default = auto)
            This variable is used to choose the type of forest that will be
            created in the Forest Inference Library. It is not required
            while using predict_model='CPU'.

             * ``'auto'`` - choose the storage type automatically
               (currently True is chosen by auto)
             * ``False`` - create a dense forest
             * ``True`` - create a sparse forest, requires algo='naive'
               or algo='auto'

        Returns
        -------
        accuracy : float
           Accuracy of the model [0.0 - 1.0]
        """

        nvtx_range_push("score RF-Classifier @randomforestclassifier.pyx")
        cdef uintptr_t X_ptr, y_ptr
        _, n_rows, _, _ = \
            input_to_cuml_array(X, check_dtype=self.dtype,
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None),
                                check_cols=self.n_cols)
        y_m, n_rows, _, y_dtype = \
            input_to_cuml_array(y, check_dtype=np.int32,
                                convert_to_dtype=(np.int32 if convert_dtype
                                                  else False))
        y_ptr = y_m.ptr
        preds = self.predict(X,
                             threshold=threshold, algo=algo,
                             convert_dtype=convert_dtype,
                             predict_model=predict_model,
                             num_classes=num_classes,
                             fil_sparse_format=fil_sparse_format)

        cdef uintptr_t preds_ptr
        preds_m, _, _, _ = \
            input_to_cuml_array(preds, convert_to_dtype=np.int32)
        preds_ptr = preds_m.ptr

        cdef handle_t* handle_ =\
            <handle_t*><uintptr_t>self.handle.getHandle()

        cdef RandomForestMetaData[float, int] *rf_forest = \
            <RandomForestMetaData[float, int]*><uintptr_t> self.rf_forest

        cdef RandomForestMetaData[double, int] *rf_forest64 = \
            <RandomForestMetaData[double, int]*><uintptr_t> self.rf_forest64

        if self.dtype == np.float32:
            self.stats = score(handle_[0],
                               rf_forest,
                               <int*> y_ptr,
                               <int> n_rows,
                               <int*> preds_ptr,
                               <int> self.verbose)
        elif self.dtype == np.float64:
            self.stats = score(handle_[0],
                               rf_forest64,
                               <int*> y_ptr,
                               <int> n_rows,
                               <int*> preds_ptr,
                               <int> self.verbose)
        else:
            raise TypeError("supports only np.float32 and np.float64 input,"
                            " but input of type '%s' passed."
                            % (str(self.dtype)))

        self.handle.sync()
        del(y_m)
        del(preds_m)
        nvtx_range_pop()
        return self.stats['accuracy']

    def get_summary_text(self):
        """
        Obtain the text summary of the random forest model
        """
        cdef RandomForestMetaData[float, int] *rf_forest = \
            <RandomForestMetaData[float, int]*><uintptr_t> self.rf_forest

        cdef RandomForestMetaData[double, int] *rf_forest64 = \
            <RandomForestMetaData[double, int]*><uintptr_t> self.rf_forest64

        if self.dtype == np.float64:
            return get_rf_summary_text(rf_forest64).decode('utf-8')
        else:
            return get_rf_summary_text(rf_forest).decode('utf-8')

    def get_detailed_text(self):
        """
        Obtain the detailed information for the random forest model, as text
        """
        cdef RandomForestMetaData[float, int] *rf_forest = \
            <RandomForestMetaData[float, int]*><uintptr_t> self.rf_forest

        cdef RandomForestMetaData[double, int] *rf_forest64 = \
            <RandomForestMetaData[double, int]*><uintptr_t> self.rf_forest64

        if self.dtype == np.float64:
            return get_rf_detailed_text(rf_forest64).decode('utf-8')
        else:
            return get_rf_detailed_text(rf_forest).decode('utf-8')

    def get_json(self):
        """
        Export the Random Forest model as a JSON string
        """
        cdef RandomForestMetaData[float, int] *rf_forest = \
            <RandomForestMetaData[float, int]*><uintptr_t> self.rf_forest

        cdef RandomForestMetaData[double, int] *rf_forest64 = \
            <RandomForestMetaData[double, int]*><uintptr_t> self.rf_forest64

        if self.dtype == np.float64:
            return get_rf_json(rf_forest64).decode('utf-8')
        return get_rf_json(rf_forest).decode('utf-8')

#
# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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
from treelite import Model as TreeliteModel

import cuml.internals
import cuml.internals.nvtx as nvtx
from cuml.common import input_to_cuml_array
from cuml.common.doc_utils import generate_docstring, insert_into_docstring
from cuml.ensemble.randomforest_common import BaseRandomForestModel
from cuml.fil.fil import ForestInference
from cuml.internals.array import CumlArray
from cuml.internals.mixins import RegressorMixin
from cuml.internals.utils import check_random_seed

from libc.stdint cimport uint64_t, uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t

from cuml.ensemble.randomforest_shared cimport *
from cuml.internals.logger cimport level_enum


cdef extern from "cuml/ensemble/randomforest.hpp" namespace "ML" nogil:

    cdef void fit(handle_t& handle,
                  RandomForestMetaData[float, float]*,
                  float*,
                  int,
                  int,
                  float*,
                  RF_params,
                  level_enum) except +

    cdef void fit(handle_t& handle,
                  RandomForestMetaData[double, double]*,
                  double*,
                  int,
                  int,
                  double*,
                  RF_params,
                  level_enum) except +

    cdef void predict(handle_t& handle,
                      RandomForestMetaData[float, float] *,
                      float*,
                      int,
                      int,
                      float*,
                      level_enum) except +

    cdef void predict(handle_t& handle,
                      RandomForestMetaData[double, double]*,
                      double*,
                      int,
                      int,
                      double*,
                      level_enum) except +

    cdef RF_metrics score(handle_t& handle,
                          RandomForestMetaData[float, float]*,
                          float*,
                          int,
                          float*,
                          level_enum) except +

    cdef RF_metrics score(handle_t& handle,
                          RandomForestMetaData[double, double]*,
                          double*,
                          int,
                          double*,
                          level_enum) except +


class RandomForestRegressor(BaseRandomForestModel,
                            RegressorMixin):

    """
    Implements a Random Forest regressor model which fits multiple decision
    trees in an ensemble.

    .. note:: Note that the underlying algorithm for tree node splits differs
      from that used in scikit-learn. By default, the cuML Random Forest uses a
      quantile-based algorithm to determine splits, rather than an exact
      count. You can tune the size of the quantiles with the `n_bins` parameter

    .. note:: You can export cuML Random Forest models and run predictions
      with them on machines without an NVIDIA GPUs. See
      https://docs.rapids.ai/api/cuml/nightly/pickling_cuml_models.html
      for more details.

    Examples
    --------

    .. code-block:: python

        >>> import cupy as cp
        >>> from cuml.ensemble import RandomForestRegressor as curfr
        >>> X = cp.asarray([[0,10],[0,20],[0,30],[0,40]], dtype=cp.float32)
        >>> y = cp.asarray([0.0,1.0,2.0,3.0], dtype=cp.float32)
        >>> cuml_model = curfr(max_features=1.0, n_bins=128,
        ...                    min_samples_leaf=1,
        ...                    min_samples_split=2,
        ...                    n_estimators=40, accuracy_metric='r2')
        >>> cuml_model.fit(X,y)
        RandomForestRegressor()
        >>> cuml_score = cuml_model.score(X,y)
        >>> print("MSE score of cuml : ", cuml_score) # doctest: +SKIP
        MSE score of cuml :  0.9076250195503235

    Parameters
    ----------
    n_estimators : int (default = 100)
        Number of trees in the forest. (Default changed to 100 in cuML 0.11)
    split_criterion : int or string (default = ``2`` (``'mse'``))
        The criterion used to split nodes.\n
         * ``0`` or ``'gini'`` for gini impurity
         * ``1`` or ``'entropy'`` for information gain (entropy)
         * ``2`` or ``'mse'`` for mean squared error
         * ``4`` or ``'poisson'`` for poisson half deviance
         * ``5`` or ``'gamma'`` for gamma half deviance
         * ``6`` or ``'inverse_gaussian'`` for inverse gaussian deviance

        ``0``, ``'gini'``, ``1`` and ``'entropy'`` not valid for regression.
    bootstrap : boolean (default = True)
        Control bootstrapping.\n
            * If ``True``, each tree in the forest is built
              on a bootstrapped sample with replacement.
            * If ``False``, the whole dataset is used to build each tree.
    max_samples : float (default = 1.0)
        Ratio of dataset rows used while fitting each tree.
    max_depth : int (default = 16)
        Maximum tree depth. Must be greater than 0.
        Unlimited depth (i.e, until leaves are pure)
        is not supported.\n
        .. note:: This default differs from scikit-learn's
          random forest, which defaults to unlimited depth.
    max_leaves : int (default = -1)
        Maximum leaf nodes per tree. Soft constraint. Unlimited,
        If ``-1``.
    max_features : {'sqrt', 'log2', None}, int or float (default = 1.0)
        The number of features to consider per node split:

        * If an int then ``max_features`` is the absolute count of features to be used.
        * If a float then ``max_features`` is used as a fraction.
        * If ``'sqrt'`` then ``max_features=1/sqrt(n_features)``.
        * If ``'log2'`` then ``max_features=log2(n_features)/n_features``.
        * If ``None`` then ``max_features=n_features``

        .. versionchanged:: 24.06
          The default of `max_features` changed from `"auto"` to 1.0.

    n_bins : int (default = 128)
        Maximum number of bins used by the split algorithm per feature.
        For large problems, particularly those with highly-skewed input data,
        increasing the number of bins may improve accuracy.
    n_streams : int (default = 4 )
        Number of parallel streams used for forest building
        For nearly reproducible results, set ``n_streams=1``. If ``n_streams``
        is greater than 1, results may vary due to unpredictable differences in
        stream/thread timing, even when a ``random_state`` is specified.
    min_samples_leaf : int or float (default = 1)
        The minimum number of samples (rows) in each leaf node.\n
         * If type ``int``, then ``min_samples_leaf`` represents the minimum
           number.\n
         * If ``float``, then ``min_samples_leaf`` represents a fraction and
           ``ceil(min_samples_leaf * n_rows)`` is the minimum number of
           samples for each leaf node.
    min_samples_split : int or float (default = 2)
        The minimum number of samples required to split an internal
        node.\n
         * If type ``int``, then min_samples_split represents the minimum
           number.
         * If type ``float``, then ``min_samples_split`` represents a fraction
           and ``max(2, ceil(min_samples_split * n_rows))`` is the minimum
           number of samples for each split.
    min_impurity_decrease : float (default = 0.0)
        The minimum decrease in impurity required for node to be split
    accuracy_metric : string (default = 'r2')
        Decides the metric used to evaluate the performance of the model.
        In the 0.16 release, the default scoring metric was changed
        from mean squared error to r-squared.\n
         * for r-squared : ``'r2'``
         * for median of abs error : ``'median_ae'``
         * for mean of abs error : ``'mean_ae'``
         * for mean square error' : ``'mse'``
    max_batch_size : int (default = 4096)
        Maximum number of nodes that can be processed in a given batch.
    random_state : int (default = None)
        Seed for the random number generator. Unseeded by default. Does not
        currently fully guarantee the exact same results.
    oob_score : bool (default = False)
        Whether to compute the out-of-bag R-squared score on the training dataset.
        Only valid when bootstrap=True.
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
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Notes
    -----
    **Known Limitations**\n
    This is an early release of the cuML
    Random Forest code. It contains a few known limitations:

      * GPU-based inference is only supported with 32-bit (float32) data-types.
        Alternatives are to use CPU-based inference for 64-bit (float64)
        data-types, or let the default automatic datatype conversion occur
        during GPU inference.

    For additional docs, see `scikitlearn's RandomForestRegressor
    <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html>`_.
    """

    _cpu_class_path = "sklearn.ensemble.RandomForestRegressor"
    RF_type = REGRESSION

    @classmethod
    def _get_param_names(cls):
        return [*super()._get_param_names(), "accuracy_metric"]

    def __init__(self, *,
                 split_criterion=2,
                 max_features=1.0,
                 accuracy_metric='r2',
                 handle=None,
                 verbose=False,
                 output_type=None,
                 **kwargs):
        self.accuracy_metric = accuracy_metric
        super().__init__(
            split_criterion=split_criterion,
            max_features=max_features,
            handle=handle,
            verbose=verbose,
            output_type=output_type,
            **kwargs)

    # TODO: Add the preprocess and postprocess functions in the cython code to
    # normalize the labels
    def __getstate__(self):
        state = self.__dict__.copy()
        cdef size_t params_t
        cdef  RandomForestMetaData[float, float] *rf_forest
        cdef  RandomForestMetaData[double, double] *rf_forest64
        cdef size_t params_t64
        if self.n_cols:
            # only if model has been fit previously
            self._serialize_treelite_bytes()  # Ensure we have this cached
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
        state["_verbose"] = self._verbose
        state["treelite_serialized_bytes"] = self.treelite_serialized_bytes
        state['handle'] = self.handle
        state["split_criterion"] = self.split_criterion

        return state

    def __setstate__(self, state):
        super(RandomForestRegressor, self).__init__(
            split_criterion=state["split_criterion"],
            handle=state["handle"], verbose=state['_verbose'])
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

        self.treelite_serialized_bytes = state["treelite_serialized_bytes"]
        self.__dict__.update(state)

    def __del__(self):
        self._reset_forest_data()

    def _reset_forest_data(self):
        """Free memory allocated by this instance and clear instance vars."""
        if hasattr(self, "rf_forest") and self.rf_forest:
            delete_rf_metadata(
                <RandomForestMetaData[float, float]*><uintptr_t>
                self.rf_forest)
            self.rf_forest = 0
        if hasattr(self, "rf_forest64") and self.rf_forest64:
            delete_rf_metadata(
                <RandomForestMetaData[double, double]*><uintptr_t>
                self.rf_forest64)
            self.rf_forest64 = 0
        self.treelite_serialized_bytes = None
        self.n_cols = None

    def convert_to_treelite_model(self):
        """
        Converts the cuML RF model to a Treelite model

        Returns
        -------
        tl_to_fil_model : treelite.Model
        """
        treelite_bytes = self._serialize_treelite_bytes()
        return TreeliteModel.deserialize_bytes(treelite_bytes)

    def convert_to_fil_model(
        self,
        layout = "depth_first",
        default_chunk_size = None,
        align_bytes = None,
    ):
        """
        Create a Forest Inference (FIL) model from the trained cuML
        Random Forest model.

        Parameters
        ----------
        layout : string (default = 'depth_first')
            Specifies the in-memory layout of nodes in FIL forests. Options:
            'depth_first', 'layered', 'breadth_first'.
        default_chunk_size : int, optional (default = None)
            Determines how batches are further subdivided for parallel processing.
            The optimal value depends on hardware, model, and batch size.
            If None, will be automatically determined.
        align_bytes : int, optional (default = None)
            If specified, trees will be padded such that their in-memory size is
            a multiple of this value. This can improve performance by guaranteeing
            that memory reads from trees begin on a cache line boundary.
            Typical values are 0 or 128 on GPU and 0 or 64 on CPU.

        Returns
        -------
        fil_model : ForestInference
            A Forest Inference model which can be used to perform
            inferencing on the random forest model.
        """
        treelite_bytes = self._serialize_treelite_bytes()
        return ForestInference(
            treelite_model=treelite_bytes,
            output_type="input",
            is_classifier=False,
            layout=layout,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
        )

    @nvtx.annotate(
        message="fit RF-Regressor @randomforestregressor.pyx",
        domain="cuml_python")
    @generate_docstring()
    @cuml.internals.api_base_return_any_skipall
    def fit(self, X, y, *, convert_dtype=True):
        """
        Perform Random Forest Regression on the input data

        """

        X_m, y_m, max_feature_val = self._dataset_setup_for_fit(X, y,
                                                                convert_dtype)

        # Reset the old tree data for new fit call
        cdef uintptr_t X_ptr, y_ptr
        X_ptr = X_m.ptr
        y_ptr = y_m.ptr

        cdef handle_t* handle_ =\
            <handle_t*><uintptr_t>self.handle.getHandle()

        cdef RandomForestMetaData[float, float] *rf_forest = \
            new RandomForestMetaData[float, float]()
        self.rf_forest = <uintptr_t> rf_forest
        cdef RandomForestMetaData[double, double] *rf_forest64 = \
            new RandomForestMetaData[double, double]()
        self.rf_forest64 = <uintptr_t> rf_forest64
        if self.random_state is None:
            seed_val = <uintptr_t>NULL
        else:
            seed_val = <uintptr_t>check_random_seed(self.random_state)

        rf_params = set_rf_params(<int> self.max_depth,
                                  <int> self.max_leaves,
                                  <float> max_feature_val,
                                  <int> self.n_bins,
                                  <int> self.min_samples_leaf,
                                  <int> self.min_samples_split,
                                  <float> self.min_impurity_decrease,
                                  <bool> self.bootstrap,
                                  <int> self.n_estimators,
                                  <float> self.max_samples,
                                  <uint64_t> seed_val,
                                  <CRITERION> self.split_criterion,
                                  <int> self.n_streams,
                                  <int> self.max_batch_size,
                                  <bool> self.oob_score,
                                  <bool> True)  # compute_feature_importance=True

        if self.dtype == np.float32:
            fit(handle_[0],
                rf_forest,
                <float*> X_ptr,
                <int> self.n_rows,
                <int> self.n_cols,
                <float*> y_ptr,
                rf_params,
                <level_enum> self.verbose)

        else:
            rf_params64 = rf_params
            fit(handle_[0],
                rf_forest64,
                <double*> X_ptr,
                <int> self.n_rows,
                <int> self.n_cols,
                <double*> y_ptr,
                rf_params64,
                <level_enum> self.verbose)
        # make sure that the `fit` is complete before the following delete
        # call happens
        self.handle.sync()
        
        # Extract OOB score and feature importances from the forest
        if self.oob_score:
            if self.dtype == np.float32:
                self._oob_score_ = get_oob_score(rf_forest)
            else:
                self._oob_score_ = get_oob_score(rf_forest64)
        
        # Extract feature importances
        cdef vector[float] fi_float
        cdef vector[double] fi_double
        if self.dtype == np.float32:
            fi_float = get_feature_importances(rf_forest)
            self._feature_importances_ = [fi_float[i] for i in range(fi_float.size())]
        else:
            fi_double = get_feature_importances(rf_forest64)
            self._feature_importances_ = [fi_double[i] for i in range(fi_double.size())]
        
        del X_m
        del y_m
        return self

    def _predict_model_on_cpu(
        self,
        X,
        convert_dtype = True,
    ) -> CumlArray:
        cdef uintptr_t X_ptr
        X_m, n_rows, n_cols, dtype = \
            input_to_cuml_array(X, order='C',
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None),
                                check_cols=self.n_cols)
        X_ptr = X_m.ptr

        preds = CumlArray.zeros(n_rows, dtype=dtype)
        cdef uintptr_t preds_ptr = preds.ptr

        cdef handle_t* handle_ = \
            <handle_t*> <uintptr_t> self.handle.getHandle()

        cdef RandomForestMetaData[float, float] *rf_forest = \
            <RandomForestMetaData[float, float] *> <uintptr_t> self.rf_forest

        cdef RandomForestMetaData[double, double] *rf_forest64 = \
            <RandomForestMetaData[double, double] *> <uintptr_t> self.rf_forest64
        if self.dtype == np.float32:
            predict(handle_[0],
                    rf_forest,
                    <float*> X_ptr,
                    <int> n_rows,
                    <int> n_cols,
                    <float*> preds_ptr,
                    <level_enum> self.verbose)

        elif self.dtype == np.float64:
            predict(handle_[0],
                    rf_forest64,
                    <double*> X_ptr,
                    <int> n_rows,
                    <int> n_cols,
                    <double*> preds_ptr,
                    <level_enum> self.verbose)
        else:
            raise TypeError("supports only float32 and float64 input,"
                            " but input of type '%s' passed."
                            % (str(self.dtype)))

        self.handle.sync()
        # synchronous w/o a stream
        del X_m
        return preds

    @nvtx.annotate(
        message="predict RF-Regressor @randomforestclassifier.pyx",
        domain="cuml_python")
    @insert_into_docstring(parameters=[('dense', '(n_samples, n_features)')],
                           return_values=[('dense', '(n_samples, 1)')])
    def predict(
        self,
        X,
        *,
        convert_dtype = True,
        predict_model = "GPU",
        layout = "depth_first",
        default_chunk_size = None,
        align_bytes = None,
    ) -> CumlArray:
        """
        Predicts the values for X.

        Parameters
        ----------
        X : {}
        convert_dtype : bool, optional (default = True)
            When set to True, the predict method will, when necessary, convert
            the input to the data type which was used to train the model. This
            will increase memory used for the method.
        predict_model : string (default = 'GPU')
            'GPU' to predict using the GPU, 'CPU' otherwise. The GPU can only
            be used if the model was trained on float32 data and `X` is float32
            or convert_dtype is set to True.
        layout : string (default = 'depth_first')
            Specifies the in-memory layout of nodes in FIL forests. Options:
            'depth_first', 'layered', 'breadth_first'. Only used when predict_model='GPU'.
        default_chunk_size : int, optional (default = None)
            Determines how batches are further subdivided for parallel processing.
            The optimal value depends on hardware, model, and batch size.
            If None, will be automatically determined. Only used when predict_model='GPU'.
        align_bytes : int, optional (default = None)
            If specified, trees will be padded such that their in-memory size is
            a multiple of this value. This can improve performance by guaranteeing
            that memory reads from trees begin on a cache line boundary.
            Typical values are 0 or 128 on GPU and 0 or 64 on CPU.
            Only used when predict_model='GPU'.

        Returns
        -------
        y : {}
        """
        if predict_model == "CPU":
            preds = self._predict_model_on_cpu(
                X=X,
                convert_dtype=convert_dtype,
            )
        else:
            preds = self._predict_model_on_gpu(
                X=X,
                is_classifier=False,
                predict_proba=False,
                convert_dtype=convert_dtype,
                layout=layout,
                default_chunk_size=default_chunk_size,
                align_bytes=align_bytes,
            )

        # Reshape to 1D array if the output would be (n, 1) to match
        # the output shape behavior of scikit-learn.
        if len(preds.shape) == 2 and preds.shape[1] == 1:
            preds = CumlArray(preds.to_output('array').reshape(-1))

        return preds

    @nvtx.annotate(
        message="score RF-Regressor @randomforestclassifier.pyx",
        domain="cuml_python")
    @insert_into_docstring(parameters=[('dense', '(n_samples, n_features)'),
                                       ('dense', '(n_samples, 1)')])
    def score(
        self,
        X,
        y,
        *,
        convert_dtype = True,
        predict_model = "GPU",
        layout = "depth_first",
        default_chunk_size = None,
        align_bytes = None,
    ):
        """
        Calculates the accuracy metric score of the model for X.
        In the 0.16 release, the default scoring metric was changed
        from mean squared error to r-squared.

        Parameters
        ----------
        X : {}
        y : {}
        convert_dtype : bool (default = True)
            When True, automatically convert the input to the data type used
            to train the model. This may increase memory usage.
        predict_model : string (default = 'GPU')
            Device to use for prediction: 'GPU' or 'CPU'.
        layout : string (default = 'depth_first')
            Specifies the in-memory layout of nodes in FIL forests. Options:
            'depth_first', 'layered', 'breadth_first'.
        default_chunk_size : int, optional (default = None)
            Determines how batches are further subdivided for parallel processing.
            The optimal value depends on hardware, model, and batch size.
            If None, will be automatically determined.
        align_bytes : int, optional (default = None)
            If specified, trees will be padded such that their in-memory size is
            a multiple of this value. This can improve performance by guaranteeing
            that memory reads from trees begin on a cache line boundary.
            Typical values are 0 or 128 on GPU and 0 or 64 on CPU.

        Returns
        -------
        mean_square_error : float or
        median_abs_error : float or
        mean_abs_error : float
        """
        from cuml.metrics.regression import r2_score

        cdef uintptr_t y_ptr
        _, n_rows, _, dtype = \
            input_to_cuml_array(X,
                                convert_to_dtype=(self.dtype if convert_dtype
                                                  else None))
        y_m, n_rows, _, _ = \
            input_to_cuml_array(y,
                                convert_to_dtype=(dtype if convert_dtype
                                                  else False))
        y_ptr = y_m.ptr
        preds = self.predict(
            X,
            convert_dtype=convert_dtype,
            predict_model=predict_model,
            layout=layout,
            default_chunk_size=default_chunk_size,
            align_bytes=align_bytes,
        )

        cdef uintptr_t preds_ptr
        preds_m, _, _, _ = \
            input_to_cuml_array(preds, convert_to_dtype=dtype)
        preds_ptr = preds_m.ptr

        # shortcut for default accuracy metric of r^2
        if self.accuracy_metric == "r2":
            stats = r2_score(y_m, preds)
            self.handle.sync()
            del y_m
            del preds_m
            return stats

        cdef handle_t* handle_ =\
            <handle_t*><uintptr_t>self.handle.getHandle()

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
                                    <level_enum> self.verbose)

        elif self.dtype == np.float64:
            self.temp_stats = score(handle_[0],
                                    rf_forest64,
                                    <double*> y_ptr,
                                    <int> n_rows,
                                    <double*> preds_ptr,
                                    <level_enum> self.verbose)

        if self.accuracy_metric == 'median_ae':
            stats = self.temp_stats['median_abs_error']
        if self.accuracy_metric == 'mean_ae':
            stats = self.temp_stats['mean_abs_error']
        else:
            stats = self.temp_stats['mean_squared_error']

        self.handle.sync()
        del y_m
        del preds_m
        return stats

    def get_summary_text(self):
        """
        Obtain the text summary of the random forest model
        """
        cdef RandomForestMetaData[float, float] *rf_forest = \
            <RandomForestMetaData[float, float]*><uintptr_t> self.rf_forest

        cdef RandomForestMetaData[double, double] *rf_forest64 = \
            <RandomForestMetaData[double, double]*><uintptr_t> self.rf_forest64

        if self.dtype == np.float64:
            return get_rf_summary_text(rf_forest64).decode('utf-8')
        else:
            return get_rf_summary_text(rf_forest).decode('utf-8')

    def get_detailed_text(self):
        """
        Obtain the detailed information for the random forest model, as text
        """
        cdef RandomForestMetaData[float, float] *rf_forest = \
            <RandomForestMetaData[float, float]*><uintptr_t> self.rf_forest

        cdef RandomForestMetaData[double, double] *rf_forest64 = \
            <RandomForestMetaData[double, double]*><uintptr_t> self.rf_forest64

        if self.dtype == np.float64:
            return get_rf_detailed_text(rf_forest64).decode('utf-8')
        else:
            return get_rf_detailed_text(rf_forest).decode('utf-8')

    def get_json(self):
        """
        Export the Random Forest model as a JSON string
        """
        cdef RandomForestMetaData[float, float] *rf_forest = \
            <RandomForestMetaData[float, float]*><uintptr_t> self.rf_forest

        cdef RandomForestMetaData[double, double] *rf_forest64 = \
            <RandomForestMetaData[double, double]*><uintptr_t> self.rf_forest64

        if self.dtype == np.float64:
            return get_rf_json(rf_forest64).decode('utf-8')
        return get_rf_json(rf_forest).decode('utf-8')

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

import warnings

from cuml.dask.common.base import DelayedPredictionMixin
from cuml.ensemble import RandomForestRegressor as cuRFR
from cuml.dask.ensemble.base import \
    BaseRandomForestModel
from cuml.dask.common.base import BaseEstimator


class RandomForestRegressor(BaseRandomForestModel, DelayedPredictionMixin,
                            BaseEstimator):
    """
    Experimental API implementing a multi-GPU Random Forest classifier
    model which fits multiple decision tree classifiers in an
    ensemble. This uses Dask to partition data over multiple GPUs
    (possibly on different nodes).

    Currently, this API makes the following assumptions:
     * The set of Dask workers used between instantiation, fit,
       and predict are all consistent
     * Training data comes in the form of cuDF dataframes or Dask Arrays
       distributed so that each worker has at least one partition.
     * The get_summary_text and get_detailed_text functions provides the \
        text representation of the forest on the worker.

    Future versions of the API will support more flexible data
    distribution and additional input types. User-facing APIs are
    expected to change in upcoming versions.

    The distributed algorithm uses an embarrassingly-parallel
    approach. For a forest with N trees being built on w workers, each
    worker simply builds N/w trees on the data it has available
    locally. In many cases, partitioning the data so that each worker
    builds trees on a subset of the total dataset works well, but
    it generally requires the data to be well-shuffled in advance.
    Alternatively, callers can replicate all of the data across
    workers so that rf.fit receives w partitions, each containing the
    same data. This would produce results approximately identical to
    single-GPU fitting.

    Please check the single-GPU implementation of Random Forest
    classifier for more information about the underlying algorithm.

    Parameters
    -----------
    n_estimators : int (default = 10)
        total number of trees in the forest (not per-worker)
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    split_algo : int (default = 1)
        0 for HIST, 1 for GLOBAL_QUANTILE
        The type of algorithm to be used to create the trees.
    split_criterion : int (default = 2)
        The criterion used to split nodes.
        0 for GINI, 1 for ENTROPY,
        2 for MSE, 3 for MAE and 4 for CRITERION_END.
        0 and 1 not valid for regression
    bootstrap : boolean (default = True)
        Control bootstrapping.
        If set, each tree in the forest is built
        on a bootstrapped sample with replacement.
        If False, the whole dataset is used to build each tree.
    bootstrap_features : boolean (default = False)
        Control bootstrapping for features.
        If features are drawn with or without replacement
    max_samples : float (default = 1.0)
        Ratio of dataset rows used while fitting each tree.
    max_depth : int (default = -1)
        Maximum tree depth. Unlimited (i.e, until leaves are pure), if -1.
    max_leaves : int (default = -1)
        Maximum leaf nodes per tree. Soft constraint. Unlimited, if -1.
    max_features : int or float or string or None (default = 'auto')
        Ratio of number of features (columns) to consider
        per node split.
        If int then max_features/n_features.
        If float then max_features is a fraction.
        If 'auto' then max_features=n_features which is 1.0.
        If 'sqrt' then max_features=1/sqrt(n_features).
        If 'log2' then max_features=log2(n_features)/n_features.
        If None, then max_features=n_features which is 1.0.
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
    accuracy_metric : string (default = 'r2')
        Decides the metric used to evaluate the performance of the model.
        In the 0.16 release, the default scoring metric was changed
        from mean squared error to r-squared.
        for r-squared : 'r2'
        for median of abs error : 'median_ae'
        for mean of abs error : 'mean_ae'
        for mean square error' : 'mse'
    n_streams : int (default = 4 )
        Number of parallel streams used for forest building
    workers : optional, list of strings
        Dask addresses of workers to use for computation.
        If None, all available Dask workers will be used.
    random_state : int (default = None)
        Seed for the random number generator. Unseeded by default.
    seed : int (default = None)
        Base seed for the random number generator. Unseeded by default. Does
        not currently fully guarantee the exact same results.

        .. deprecated:: 0.15
           Parameter `seed` is deprecated and will be removed in 0.17. Please
           use `random_state` instead

    ignore_empty_partitions: Boolean (default = False)
        Specify behavior when a worker does not hold any data
        while splitting. When True, it returns the results from workers
        with data (the number of trained estimators will be less than
        n_estimators) When False, throws a RuntimeError.
        This is an experiemental parameter, and may be removed
        in the future.

    """

    def __init__(
        self,
        workers=None,
        client=None,
        verbose=False,
        n_estimators=10,
        random_state=None,
        seed=None,
        ignore_empty_partitions=False,
        **kwargs
    ):
        super(RandomForestRegressor, self).__init__(client=client,
                                                    verbose=verbose,
                                                    **kwargs)

        if seed is not None:
            if random_state is None:
                warnings.warn("Parameter 'seed' is deprecated and will be"
                              " removed in 0.17. Please use 'random_state'"
                              " instead. Setting 'random_state' as the"
                              " curent 'seed' value",
                              DeprecationWarning)
                random_state = seed
            else:
                warnings.warn("Both 'seed' and 'random_state' parameters were"
                              " set. Using 'random_state' since 'seed' is"
                              " deprecated and will be removed in 0.17.",
                              DeprecationWarning)

        self._create_model(
            model_func=RandomForestRegressor._construct_rf,
            client=client,
            workers=workers,
            n_estimators=n_estimators,
            base_seed=random_state,
            ignore_empty_partitions=ignore_empty_partitions,
            **kwargs
        )

    @staticmethod
    def _construct_rf(
        n_estimators,
        random_state,
        **kwargs
    ):
        return cuRFR(
            n_estimators=n_estimators,
            random_state=random_state,
            **kwargs)

    @staticmethod
    def _predict_model_on_cpu(model, X, convert_dtype):
        return model._predict_model_on_cpu(X, convert_dtype=convert_dtype)

    def get_summary_text(self):
        """
        Obtain the text summary of the random forest model
        """
        return self._get_summary_text()

    def get_detailed_text(self):
        """
        Obtain the detailed information for the random forest model, as text
        """
        return self._get_detailed_text()

    def get_json(self):
        """
        Export the Random Forest model as a JSON string
        """
        return self._get_json()

    def fit(self, X, y, convert_dtype=False):
        """
        Fit the input data with a Random Forest regression model

        IMPORTANT: X is expected to be partitioned with at least one partition
        on each Dask worker being used by the forest (self.workers).

        When persisting data, you can use
        `cuml.dask.common.utils.persist_across_workers` to simplify this:

        .. code-block:: python

            X_dask_cudf = dask_cudf.from_cudf(X_cudf, npartitions=n_workers)
            y_dask_cudf = dask_cudf.from_cudf(y_cudf, npartitions=n_workers)
            X_dask_cudf, y_dask_cudf = persist_across_workers(dask_client,
                                                              [X_dask_cudf,
                                                               y_dask_cudf])

        This is equivalent to calling `persist` with the data and workers):

        .. code-block:: python

            X_dask_cudf, y_dask_cudf = dask_client.persist([X_dask_cudf,
                                                            y_dask_cudf],
                                                           workers={
                                                           X_dask_cudf=workers,
                                                           y_dask_cudf=workers
                                                           })

        Parameters
        ----------
        X : Dask cuDF dataframe  or CuPy backed Dask Array (n_rows, n_features)
            Distributed dense matrix (floats or doubles) of shape
            (n_samples, n_features).
        y : Dask cuDF dataframe  or CuPy backed Dask Array (n_rows, 1)
            Labels of training examples.
            **y must be partitioned the same way as X**
        convert_dtype : bool, optional (default = False)
            When set to True, the fit method will, when necessary, convert
            y to be the same data type as X if they differ. This will increase
            memory used for the method.

        """
        self.internal_model = None
        self._fit(model=self.rfs,
                  dataset=(X, y),
                  convert_dtype=convert_dtype)
        return self

    def predict(self, X, predict_model="GPU", algo='auto',
                convert_dtype=True, fil_sparse_format='auto',
                delayed=True):
        """
        Predicts the regressor outputs for X.


        GPU-based prediction in a multi-node, multi-GPU context works
        by sending the sub-forest from each worker to the client,
        concatenating these into one forest with the full
        `n_estimators` set of trees, and sending this combined forest to
        the workers, which will each infer on their local set of data.
        This allows inference to scale to large datasets, but the forest
        transmission incurs overheads for very large trees. For inference
        on small datasets, this overhead may dominate prediction time.
        Within the worker, this uses the cuML Forest Inference Library
        (cuml.fil) for high-throughput prediction.

        The 'CPU' fallback method works with sub-forests in-place,
        broadcasting the datasets to all workers and combining predictions
        via an averaging method at the end. This method is slower
        on a per-row basis but may be faster for problems with many trees
        and few rows.

        In the 0.15 cuML release, inference will be updated with much
        faster tree transfer.

        Parameters
        ----------
        X : Dask cuDF dataframe  or CuPy backed Dask Array (n_rows, n_features)
            Distributed dense matrix (floats or doubles) of shape
            (n_samples, n_features).
        algo : string (default = 'auto')
            This is optional and required only while performing the
            predict operation on the GPU.
            'naive' - simple inference using shared memory
            'tree_reorg' - similar to naive but trees rearranged to be more
            coalescing-friendly
            'batch_tree_reorg' - similar to tree_reorg but predicting
            multiple rows per thread block
            `algo` - choose the algorithm automatically. Currently
            'batch_tree_reorg' is used for dense storage
            and 'naive' for sparse storage
        convert_dtype : bool, optional (default = True)
            When set to True, the predict method will, when necessary, convert
            the input to the data type which was used to train the model. This
            will increase memory used for the method.
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
        delayed : bool (default = True)
            Whether to do a lazy prediction (and return Delayed objects) or an
            eagerly executed one.

        Returns
        -------
        y : Dask cuDF dataframe or CuPy backed Dask Array (n_rows, 1)

        """
        if predict_model == "CPU":
            preds = self.predict_model_on_cpu(X, convert_dtype=convert_dtype)

        else:
            preds = \
                self._predict_using_fil(X,
                                        algo=algo,
                                        convert_dtype=convert_dtype,
                                        fil_sparse_format=fil_sparse_format,
                                        delayed=delayed)
        return preds

    def predict_using_fil(self, X, delayed, **kwargs):
        if self._get_internal_model() is None:
            self._set_internal_model(self._concat_treelite_models())
        return self._predict_using_fil(X=X,
                                       delayed=delayed,
                                       **kwargs)

    """
    TODO : Update function names used for CPU predict.
           Cuml issue #1854 has been created to track this.
    """
    def predict_model_on_cpu(self, X, convert_dtype):
        workers = self.workers

        X_Scattered = self.client.scatter(X)

        futures = list()
        for n, w in enumerate(workers):
            futures.append(
                self.client.submit(
                    RandomForestRegressor._predict_model_on_cpu,
                    self.rfs[w],
                    X_Scattered,
                    convert_dtype,
                    workers=[w],
                )
            )

        rslts = self.client.gather(futures, errors="raise")
        pred = list()

        for i in range(len(X)):
            pred_per_worker = 0.0
            for d in range(len(rslts)):
                pred_per_worker = pred_per_worker + rslts[d][i]

            pred.append(pred_per_worker / len(rslts))

        return pred

    def get_params(self, deep=True):
        """
        Returns the value of all parameters
        required to configure this estimator as a dictionary.

        Parameters
        -----------
        deep : boolean (default = True)
        """
        return self._get_params(deep)

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

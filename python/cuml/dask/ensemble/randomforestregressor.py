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

from cuml.dask.common import raise_exception_from_futures
from cuml.dask.common.base import DelayedPredictionMixin
from cuml.ensemble import RandomForestRegressor as cuRFR
from cuml.dask.ensemble.randomforestcommon import \
    BaseRandomForestModel

from dask.distributed import default_client, wait


class RandomForestRegressor(BaseRandomForestModel, DelayedPredictionMixin):
    """
    Experimental API implementing a multi-GPU Random Forest classifier
    model which fits multiple decision tree classifiers in an
    ensemble. This uses Dask to partition data over multiple GPUs
    (possibly on different nodes).

    Currently, this API makes the following assumptions:
    * The set of Dask workers used between instantiation, fit,
    and predict are all consistent
    * Training data is comes in the form of cuDF dataframes,
    distributed so that each worker has at least one partition.

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
        If it is None, a new one is created just for this class.
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
        If false, sampling without replacement is done.
    bootstrap_features : boolean (default = False)
        Control bootstrapping for features.
        If features are drawn with or without replacement
    rows_sample : float (default = 1.0)
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
    min_rows_per_node : int or float (default = 2)
        The minimum number of samples (rows) needed to split a node.
        If int then number of sample rows
        If float the min_rows_per_sample*n_rows
    accuracy_metric : string (default = 'mse')
        Decides the metric used to evaluate the performance of the model.
        for median of abs error : 'median_ae'
        for mean of abs error : 'mean_ae'
        for mean square error' : 'mse'
    n_streams : int (default = 4 )
        Number of parallel streams used for forest building
    workers : optional, list of strings
        Dask addresses of workers to use for computation.
        If None, all available Dask workers will be used.

    """

    def __init__(
        self,
        n_estimators=10,
        max_depth=-1,
        max_features="auto",
        n_bins=8,
        split_algo=1,
        split_criterion=2,
        bootstrap=True,
        bootstrap_features=False,
        verbose=False,
        min_rows_per_node=2,
        rows_sample=1.0,
        max_leaves=-1,
        n_streams=4,
        accuracy_metric="mse",
        dtype=None,
        min_samples_leaf=None,
        min_weight_fraction_leaf=None,
        n_jobs=None,
        max_leaf_nodes=None,
        min_impurity_decrease=None,
        min_impurity_split=None,
        oob_score=None,
        random_state=None,
        warm_start=None,
        class_weight=None,
        quantile_per_tree=False,
        criterion=None,
        workers=None,
        client=None
    ):

        unsupported_sklearn_params = {
            "criterion": criterion,
            "min_samples_leaf": min_samples_leaf,
            "min_weight_fraction_leaf": min_weight_fraction_leaf,
            "max_leaf_nodes": max_leaf_nodes,
            "min_impurity_decrease": min_impurity_decrease,
            "min_impurity_split": min_impurity_split,
            "oob_score": oob_score,
            "n_jobs": n_jobs,
            "random_state": random_state,
            "warm_start": warm_start,
            "class_weight": class_weight,
        }

        for key, vals in unsupported_sklearn_params.items():
            if vals is not None:
                raise TypeError(
                    " The Scikit-learn variable ",
                    key,
                    " is not supported in cuML,"
                    " please read the cuML documentation for"
                    " more information",
                )

        self.n_estimators = n_estimators
        self.n_estimators_per_worker = list()

        self.client = default_client() if client is None else client
        if workers is None:
            workers = self.client.has_what().keys()
        self.workers = workers
        self._create_the_model(
            model_func=RandomForestRegressor._func_build_rf,
            max_depth=max_depth,
            n_streams=n_streams,
            max_features=max_features,
            n_bins=n_bins,
            split_algo=split_algo,
            split_criterion=split_criterion,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            verbose=verbose,
            min_rows_per_node=min_rows_per_node,
            rows_sample=rows_sample,
            max_leaves=max_leaves,
            accuracy_metric=accuracy_metric,
            quantile_per_tree=quantile_per_tree,
            dtype=dtype)

    @staticmethod
    def _func_build_rf(
        n_estimators,
        seed,
        **kwargs
    ):
        return cuRFR(
            n_estimators=n_estimators,
            seed=seed,
            **kwargs)

    @staticmethod
    def _predict_model_on_cpu(model, X, convert_dtype):
        return model._predict_model_on_cpu(X, convert_dtype=convert_dtype)

    def print_summary(self):
        """
        Print the summary of the forest used to train and test the model.
        """
        return self._print_summary()

    def fit(self, X, y, convert_dtype=False):
        """
        Fit the input data with a Random Forest regression model

        IMPORTANT: X is expected to be partitioned with at least one partition
        on each Dask worker being used by the forest (self.workers).

        When persisting data, you can use
        cuml.dask.common.utils.persist_across_workers to simplify this::

            X_dask_cudf = dask_cudf.from_cudf(X_cudf, npartitions=n_workers)
            y_dask_cudf = dask_cudf.from_cudf(y_cudf, npartitions=n_workers)
            X_dask_cudf, y_dask_cudf = persist_across_workers(dask_client,
                                                              [X_dask_cudf,
                                                               y_dask_cudf])

        (this is equivalent to calling `persist` with the data and workers)::
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
            y to be the same data type as X if they differ. This
            will increase memory used for the method.
        """
        self._fit(model=self.rfs,
                  dataset=(X, y),
                  convert_dtype=convert_dtype)
        return self

    def predict(self, X, predict_model="GPU", algo='auto',
                convert_dtype=True, fil_sparse_format='auto',
                delayed=True):
        """
        Predicts the regressor outputs for X.

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
        ----------
        y : Dask cuDF dataframe  or CuPy backed Dask Array (n_rows, 1)
        """
        if predict_model == "CPU":
            preds = self.predict_model_on_cpu(X, convert_dtype=convert_dtype)

        else:
            preds = \
                self.predict_using_fil(X, predict_model=predict_model,
                                       algo=algo,
                                       convert_dtype=convert_dtype,
                                       fil_sparse_format=fil_sparse_format,
                                       delayed=delayed)
        return preds

    def predict_using_fil(self, X, delayed, **kwargs):
        self.local_model = self._concat_treelite_models()
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

        wait(futures)
        raise_exception_from_futures(futures)

        indexes = list()
        rslts = list()
        for d in range(len(futures)):
            rslts.append(futures[d].result())
            indexes.append(0)

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
        return self._get_params(model_type=RandomForestRegressor,
                                deep=True)

    def set_params(self, **params):
        """
        Sets the value of parameters required to
        configure this estimator, it functions similar to
        the sklearn set_params.

        Parameters
        -----------
        params : dict of new params
        """
        return self._set_params(model_type=RandomForestRegressor,
                                **params)

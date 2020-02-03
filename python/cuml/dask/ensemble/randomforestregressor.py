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

import cudf

from cuml.ensemble import RandomForestRegressor as cuRFR
from cuml.dask.common import extract_ddf_partitions, \
    raise_exception_from_futures, workers_to_parts

from dask.distributed import default_client, wait

import math
import random
from uuid import uuid1


class RandomForestRegressor:
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
    split_criterion: int (default = 2)
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
                Maximum tree depth. Unlimited (i.e, until leaves are pure),
                if -1.
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
    accuracy_metric : string (default = 'mse')
                      Decides the metric used to evaluate the performance
                      of the model.
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
        workers=None
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

        c = default_client()
        if workers is None:
            workers = c.has_what().keys()
        self.workers = workers
        n_workers = len(workers)
        if n_estimators < n_workers:
            raise ValueError(
                "n_estimators cannot be lower than number of dask workers."
            )

        n_est_per_worker = math.floor(n_estimators / n_workers)

        for i in range(n_workers):
            self.n_estimators_per_worker.append(n_est_per_worker)

        remaining_est = n_estimators - (n_est_per_worker * n_workers)

        for i in range(remaining_est):
            self.n_estimators_per_worker[i] = (
                self.n_estimators_per_worker[i] + 1
            )

        seeds = list()
        seeds.append(0)
        for i in range(1, len(self.n_estimators_per_worker)):
            sd = self.n_estimators_per_worker[i-1] + seeds[i-1]
            seeds.append(sd)

        key = str(uuid1())
        self.rfs = {
            worker: c.submit(
                RandomForestRegressor._func_build_rf,
                self.n_estimators_per_worker[n],
                max_depth,
                n_streams,
                max_features,
                n_bins,
                split_algo,
                split_criterion,
                bootstrap,
                bootstrap_features,
                verbose,
                min_rows_per_node,
                rows_sample,
                max_leaves,
                accuracy_metric,
                quantile_per_tree,
                seeds[n],
                key="%s-%s" % (key, n),
                workers=[worker],
            )
            for n, worker in enumerate(workers)
        }

        rfs_wait = list()
        for r in self.rfs.values():
            rfs_wait.append(r)

        wait(rfs_wait)
        raise_exception_from_futures(rfs_wait)

    @staticmethod
    def _func_build_rf(
        n_estimators,
        max_depth,
        n_streams,
        max_features,
        n_bins,
        split_algo,
        split_criterion,
        bootstrap,
        bootstrap_features,
        verbose,
        min_rows_per_node,
        rows_sample,
        max_leaves,
        accuracy_metric,
        quantile_per_tree,
        seed,
    ):

        return cuRFR(
            n_estimators=n_estimators,
            max_depth=max_depth,
            handle=None,
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
            n_streams=n_streams,
            accuracy_metric=accuracy_metric,
            quantile_per_tree=quantile_per_tree,
            seed=seed,
        )

    @staticmethod
    def _fit(model, X_df_list, y_df_list, r):
        if len(X_df_list) != len(y_df_list):
            raise ValueError("X (%d) and y (%d) partition list sizes unequal" %
                             len(X_df_list), len(y_df_list))
        if len(X_df_list) == 1:
            X_df = X_df_list[0]
            y_df = y_df_list[0]
        else:
            X_df = cudf.concat(X_df_list)
            y_df = cudf.concat(y_df_list)
        return model.fit(X_df, y_df)

    @staticmethod
    def _predict(model, X, r):
        return model.predict(X).copy_to_host()

    @staticmethod
    def _read_mod_handles(model, mod_handles):
        return model._read_mod_handles(mod_handles=mod_handles)

    @staticmethod
    def _tl_model_handles(model, model_bytes):
        return model._tl_model_handles(model_bytes=model_bytes)

    @staticmethod
    def _print_summary(model):
        model.print_summary()

    def print_summary(self):
        """
        prints the summary of the forest used to train and test the model
        """
        c = default_client()
        futures = list()
        workers = self.workers

        for n, w in enumerate(workers):
            futures.append(
                c.submit(
                    RandomForestRegressor._print_summary,
                    self.rfs[w],
                    workers=[w],
                )
            )

        wait(futures)
        raise_exception_from_futures(futures)
        return self

    def convert_to_treelite(self):
        """
        prints the summary of the forest used to train and test the model
        """
        mod_bytes = []
        size_of_mod_bytes_read = []
        for w in self.workers:
            mod_bytes.append(self.rfs[w].result().model_pbuf_bytes)

        worker_numb = [i for i in self.workers]

        list_mod_handles = []
        model = self.rfs[worker_numb[0]].result()
        for n in range(len(self.workers)):
            list_mod_handles.append(model._tl_model_handles(mod_bytes[n]))
            size_of_mod_bytes_read.append(len(mod_bytes[n]))

        return list_mod_handles, size_of_mod_bytes_read

    def check_treelite_handles(self):

        list_mod_handles, size_of_mod_bytes_read = self.convert_to_treelite()
        check_model_bytes = []
        worker_numb = [i for i in self.workers]

        model = self.rfs[worker_numb[0]].result()

        for n in range(len(self.workers)):
            check_model_bytes.append(model._read_mod_handles(
                                         list_mod_handles[n]))

        for i in range(len(size_of_mod_bytes_read)):
            check_size_of_mod_bytes_read = len(check_model_bytes[i])
            if check_size_of_mod_bytes_read != size_of_mod_bytes_read[i]:
                raise ValueError("The treelite handle obtained from each user"
                                 " are not right")

    def fit(self, X, y):
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
        X : dask_cudf.Dataframe
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Features of training examples.

        y : dask_cudf.Dataframe
            Dense matrix (floats or doubles) of shape (n_samples, 1)
            Labels of training examples.
            y must be partitioned the same way as X
        """
        c = default_client()

        X_futures = workers_to_parts(c.sync(extract_ddf_partitions, X))
        y_futures = workers_to_parts(c.sync(extract_ddf_partitions, y))

        X_partition_workers = [w for w, xc in X_futures.items()]
        y_partition_workers = [w for w, xc in y_futures.items()]

        if set(X_partition_workers) != set(self.workers) or \
           set(y_partition_workers) != set(self.workers):
            raise ValueError("""
              X is not partitioned on the same workers expected by RF\n
              X workers: %s\n
              y workers: %s\n
              RF workers: %s
            """ % (str(X_partition_workers),
                   str(y_partition_workers),
                   str(self.workers)))

        futures = list()
        for w, xc in X_futures.items():
            futures.append(
                c.submit(
                    RandomForestRegressor._fit,
                    self.rfs[w],
                    xc,
                    y_futures[w],
                    random.random(),
                    workers=[w],
                )
            )

        wait(futures)
        raise_exception_from_futures(futures)

        return self

    def predict(self, X):
        """
        Predicts the regressor outputs for X.

        Parameters
        ----------
        X : Dense matrix (floats or doubles) of shape (n_samples, n_features).

        Returns
        ----------
        y: NumPy
           Dense vector (float) of shape (n_samples, 1)

        """
        c = default_client()
        workers = self.workers

        X_Scattered = c.scatter(X)

        futures = list()
        for n, w in enumerate(workers):
            futures.append(
                c.submit(
                    RandomForestRegressor._predict,
                    self.rfs[w],
                    X_Scattered,
                    random.random(),
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
                raise ValueError("Invalid parameter for estimator")
            else:
                setattr(self, key, value)

        return self

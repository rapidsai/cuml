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

from cuml.ensemble import RandomForestRegressor as cuRFR
from cuml.dask.common import extract_ddf_partitions

from dask.distributed import default_client, wait

import math
import random


class RandomForestRegressor:
    """
    Implements a multi-GPU Random Forest regressor model which
    fits multiple decision tree classifiers in an ensemble.

    Please check the single-GPU implementation of Random Forest
    regressor for more information about the algorithm.

     Parameters
    -----------
    n_estimators : int (default = 10)
                   number of trees in the forest.
    handle : cuml.Handle
             If it is None, a new one is created just for this class.
    split_algo : int (default = 1)
                 0 for HIST, 1 for GLOBAL_QUANTILE and 2 for SPLIT_ALGO_END
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
    """

    def __init__(
        self,
        n_estimators=10,
        max_depth=-1,
        handle=None,
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
        workers = c.has_what().keys()

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

        ws = list(zip(workers, list(range(len(workers)))))

        self.rfs = {
            worker: c.submit(
                RandomForestRegressor._func_build_rf,
                n,
                self.n_estimators_per_worker[n],
                max_depth,
                handle,
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
                n_streams,
                accuracy_metric,
                quantile_per_tree,
                random.random(),
                workers=[worker],
            )
            for worker, n in ws
        }

        rfs_wait = list()
        for r in self.rfs.values():
            rfs_wait.append(r)

        wait(rfs_wait)

    @staticmethod
    def _func_build_rf(
        n,
        n_estimators,
        max_depth,
        handle,
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
        n_streams,
        accuracy_metric,
        quantile_per_tree,
        r,
    ):

        return cuRFR(
            n_estimators=n_estimators,
            max_depth=max_depth,
            handle=handle,
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
        )

    @staticmethod
    def _fit(model, X_df, y_df, r):
        return model.fit(X_df, y_df)

    @staticmethod
    def _predict(model, X, r):
        return model.predict(X)

    def fit(self, X, y):
        """
        Fit the input data to Random Forest classifier

        Parameters
        ----------
        X : dask_cudf.Dataframe
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Features of training examples.
        y : dask_cudf.Dataframe
            Dense matrix (floats or doubles) of shape (n_samples, 1)
            Labels of training examples.
        """
        c = default_client()

        X_futures = c.sync(extract_ddf_partitions, X)
        y_futures = dict(c.sync(extract_ddf_partitions, y))

        f = list()
        for w, xc in X_futures:
            f.append(
                c.submit(
                    RandomForestRegressor._fit,
                    self.rfs[w],
                    xc,
                    y_futures[w],
                    random.random(),
                    workers=[w],
                )
            )

        wait(f)

        return self

    def predict(self, X):
        """
        Predicts the labels for X.

        Parameters
        ----------
        X : dask_cudf.Dataframe
            Dense matrix (floats or doubles) of shape (n_samples, n_features).

        Returns
        ----------
        y: NumPy
           Dense vector (int) of shape (n_samples, 1)

        """
        c = default_client()
        workers = c.has_what().keys()
        ws = list(zip(workers, list(range(len(workers)))))

        X_Scattered = c.scatter(X)

        f = list()
        for w, n in ws:
            f.append(
                c.submit(
                    RandomForestRegressor._predict,
                    self.rfs[w],
                    X_Scattered,
                    random.random(),
                    workers=[w],
                )
            )

        wait(f)

        indexes = list()
        rslts = list()
        for d in range(len(f)):
            rslts.append(f[d].result())
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

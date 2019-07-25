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

from cuml.common.handle import Handle
from cuml.dask.common import extract_ddf_partitions
from cuml.dask.common.utils import parse_host_port
from cuml.ensemble import RandomForestClassifier as cuRFC

from dask import delayed
from dask.distributed import Client, default_client, get_worker, wait
import dask.dataframe as dd

import numba.cuda
import math
import random

from tornado import gen
from toolz import first

@gen.coroutine
def _extract_ddf_partitions(ddf):
    """
    Given a Dask cuDF, return a tuple with (worker, future) for each partition
    """
    client = default_client()
    
    delayed_ddf = ddf.to_delayed()
    parts = client.compute(delayed_ddf)
    yield wait(parts)
    
    key_to_part_dict = dict([(str(part.key), part) for part in parts])
    who_has = yield client.who_has(parts)

    worker_map = []
    for key, workers in who_has.items():
        worker = parse_host_port(first(workers))
        worker_map.append((worker, key_to_part_dict[key]))

    gpu_data = {worker:part for worker, part in worker_map}

    yield wait(gpu_data)

    raise gen.Return(gpu_data)


class RandomForestClassifier:
    """
    Implements a multi-GPU Random Forest classifier model which fits multiple decision
    tree classifiers in an ensemble.

    Please check the single-GPU implementation of Random Forest classifier for more information
    about the algorithm.

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
    
    def __init__(self, n_estimators=10, max_depth=-1, handle=None,
                 max_features=1.0, n_bins=8,
                 split_algo=0, split_criterion=0, min_rows_per_node=2,
                 bootstrap=True, bootstrap_features=False,
                 type_model="classifier", verbose=False,
                 rows_sample=1.0, max_leaves=-1, quantile_per_tree=False,
                 dtype=None, criterion=None,
                 min_samples_leaf=None, min_weight_fraction_leaf=None,
                 max_leaf_nodes=None, min_impurity_decrease=None,
                 min_impurity_split=None, oob_score=None, n_jobs=None,
                 random_state=None, warm_start=None, class_weight=None):


        unsupported_sklearn_params = {"criterion": criterion,
                          "min_samples_leaf": min_samples_leaf,
                          "min_weight_fraction_leaf": min_weight_fraction_leaf,
                          "max_leaf_nodes": max_leaf_nodes,
                          "min_impurity_decrease": min_impurity_decrease,
                          "min_impurity_split": min_impurity_split,
                          "oob_score": oob_score, "n_jobs": n_jobs,
                          "random_state": random_state,
                          "warm_start": warm_start,
                          "class_weight": class_weight}
                

        for key, vals in unsupported_sklearn_params.items():
            if vals is not None:
                raise TypeError("The Scikit-learn variable", key,
                                " is not supported in cuML,"
                                " please read the cuML documentation for"
                                " more information")
      
        self.n_estimators = n_estimators
        self.n_estimators_per_worker = list()
        
        c = default_client()
        workers = c.has_what().keys()
        
        n_workers = len(workers)
        if n_estimators < n_workers:
            raise ValueError('n_estimators cannot be lower than number of dask workers.')
        
        n_est_per_worker = math.floor(n_estimators / n_workers)
                    
        for i in range(n_workers):
            self.n_estimators_per_worker.append(n_est_per_worker)
            
        remaining_est = n_estimators - (n_est_per_worker * n_workers)
                
        for i in range(remaining_est):
            self.n_estimators_per_worker[i] = self.n_estimators_per_worker[i] + 1
                    
        ws = list(zip(workers, list(range(len(workers)))))
                
        self.rfs = {parse_host_port(worker):c.submit(RandomForestClassifier._func_build_rf, 
                             n, self.n_estimators_per_worker[n], 
                             max_depth, handle,
                             max_features, n_bins,
                             split_algo, split_criterion, 
                             min_rows_per_node,
                             bootstrap, bootstrap_features,
                             type_model, verbose,
                             rows_sample, max_leaves, quantile_per_tree,
                             dtype, random.random(),
                             workers=[worker])
            for worker, n in ws}
        
        rfs_wait = list()
        for r in self.rfs.values():
            rfs_wait.append(r)
                
        wait(rfs_wait)
        
        
    @staticmethod
    def _func_build_rf(n, n_estimators, max_depth, handle,
                             max_features, n_bins,
                             split_algo, split_criterion, min_rows_per_node,
                             bootstrap, bootstrap_features,
                             type_model, verbose,
                             rows_sample, max_leaves, quantile_per_tree,
                             dtype, r):
        
        return cuRFC(n_estimators=n_estimators, max_depth=max_depth, handle=handle,
                  max_features=max_features, n_bins=n_bins,
                  split_algo=split_algo, split_criterion=split_criterion, min_rows_per_node=min_rows_per_node,
                  bootstrap=bootstrap, bootstrap_features=bootstrap_features,
                  type_model=type_model, verbose=verbose,
                  rows_sample=rows_sample, max_leaves=max_leaves, quantile_per_tree=quantile_per_tree,
                  gdf_datatype=dtype)
                
    
    @staticmethod
    def _fit(model, X_df, y_df, r): 
        return model.fit(X_df, y_df)
    
    @staticmethod
    def _predict(model, X, r): 
        return model._predict_get_all(X)
    
    def fit(self, X, y):
        """
        Fit the input data to Random Forest classifier

        Parameters
        ----------
        X : Acceptable format: dask-cudf. Dense matrix (floats or doubles) of shape (n_samples, n_features).
        y : Acceptable format: dask-cudf. Dense matrix (floats or doubles) of shape (n_samples, 1)
        """
        c = default_client()

        X_futures = c.sync(_extract_ddf_partitions, X)
        y_futures = c.sync(_extract_ddf_partitions, y)
                                       
        f = list()
        for w, xc in X_futures.items():     
            f.append(c.submit(RandomForestClassifier._fit, self.rfs[w], xc, y_futures[w], random.random(),
                             workers=[w]))            
                               
        wait(f)
        
        return self
    
    def predict(self, X):              
        """
        Predicts the labels for X.

        Parameters
        ----------
        X : Acceptable format: dask-cudf. Dense matrix (floats or doubles) of shape (n_samples, n_features).

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
            f.append(c.submit(RandomForestClassifier._predict, self.rfs[parse_host_port(w)], X_Scattered, random.random(),
                             workers=[w]))
                    
        wait(f)

        indexes = list()
        rslts = list()
        for d in range(len(f)):   
            rslts.append(f[d].result())
            indexes.append(0)
                    
        pred = list()
                
        for i in range(len(X)):
            classes = dict()
            max_class = -1
            max_val = 0
            
            for d in range(len(rslts)):               
                for j in range(self.n_estimators_per_worker[d]):
                    sub_ind = indexes[d] + j
                    cls = rslts[d][sub_ind]
                    if cls not in classes.keys():
                        classes[cls] = 1
                    else:
                        classes[cls] = classes[cls] + 1

                    if classes[cls] > max_val:
                        max_val = classes[cls]
                        max_class = cls

                indexes[d] = indexes[d] + self.n_estimators_per_worker[d]

            pred.append(max_class)
            
        
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

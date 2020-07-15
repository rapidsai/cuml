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

import numpy as np

from cuml.dask.common.base import BaseEstimator
from cuml.ensemble import RandomForestClassifier as cuRFC
from cuml.dask.common.input_utils import DistributedDataHandler
from cuml.dask.common.base import DelayedPredictionMixin, \
    DelayedPredictionProbaMixin
from cuml.dask.ensemble.base import \
    BaseRandomForestModel
from dask.distributed import default_client


class RandomForestClassifier(BaseRandomForestModel, DelayedPredictionMixin,
                             DelayedPredictionProbaMixin, BaseEstimator):

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
    * The print_summary and print_detailed functions print the
    information of the forest on the worker.

    Future versions of the API will support more flexible data
    distribution and additional input types.

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
    split_criterion : The criterion used to split nodes.
        0 for GINI, 1 for ENTROPY, 4 for CRITERION_END.
        2 and 3 not valid for classification
        (default = 0)
    split_algo : 0 for HIST and 1 for GLOBAL_QUANTILE
        (default = 1)
        the algorithm to determine how nodes are split in the tree.
    split_criterion : The criterion used to split nodes.
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
        Maximum tree depth. Unlimited (i.e, until leaves are pure), if -1.
    max_leaves : int (default = -1)
        Maximum leaf nodes per tree. Soft constraint. Unlimited, if -1.
    max_features : float (default = 'auto')
        Ratio of number of features (columns) to consider
        per node split.
    n_bins : int (default = 8)
        Number of bins used by the split algorithm.
    min_rows_per_node : int (default = 2)
        The minimum number of samples (rows) needed to split a node.
    quantile_per_tree : boolean (default = False)
        Whether quantile is computed for individual RF trees.
        Only relevant for GLOBAL_QUANTILE split_algo.
    n_streams : int (default = 4 )
        Number of parallel streams used for forest building
    workers : optional, list of strings
        Dask addresses of workers to use for computation.
        If None, all available Dask workers will be used.
    seed : int (default = None)
        Base seed for the random number generator. Unseeded by default. Does
        not currently fully guarantee the exact same results.

    Examples
    ---------
    For usage examples, please see the RAPIDS notebooks repository:
    https://github.com/rapidsai/notebooks/blob/branch-0.12/cuml/random_forest_mnmg_demo.ipynb
    """

    def __init__(
        self,
        workers=None,
        client=None,
        verbose=False,
        n_estimators=10,
        seed=None,
        **kwargs
    ):

        super(RandomForestClassifier, self).__init__(client=client,
                                                     verbose=verbose,
                                                     **kwargs)

        self._create_model(
            model_func=RandomForestClassifier._construct_rf,
            client=client,
            workers=workers,
            n_estimators=n_estimators,
            base_seed=seed,
            **kwargs)

    @staticmethod
    def _construct_rf(
        n_estimators,
        seed,
        **kwargs
    ):
        return cuRFC(
            n_estimators=n_estimators,
            seed=seed,
            **kwargs
        )

    @staticmethod
    def _predict_model_on_cpu(model, X, convert_dtype):
        return model._predict_get_all(X, convert_dtype)

    def print_summary(self):
        """
        Print the summary of the forest used to train the model
        on each worker. This information is displayed on the
        individual workers and not the client.
        """
        return self._print_summary()

    def print_detailed(self):
        """
        Print detailed information of the forest used to train
        the model on each worker. This information is displayed on the
        workers and not the client.
        """
        return self._print_detailed()

    def fit(self, X, y, convert_dtype=False):
        """
        Fit the input data with a Random Forest classifier

        IMPORTANT: X is expected to be partitioned with at least one partition
        on each Dask worker being used by the forest (self.workers).

        If a worker has multiple data partitions, they will be concatenated
        before fitting, which will lead to additional memory usage. To minimize
        memory consumption, ensure that each worker has exactly one partition.

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
            y to be of dtype int32. This will increase memory used for
            the method.
        """
        self.num_classes = len(y.unique())
        self._set_internal_model(None)
        self._fit(model=self.rfs,
                  dataset=(X, y),
                  convert_dtype=convert_dtype)
        return self

    def predict(self, X, output_class=True, algo='auto', threshold=0.5,
                convert_dtype=True, predict_model="GPU",
                fil_sparse_format='auto', delayed=True):
        """
        Predicts the labels for X.

        GPU-based prediction in a multi-node, multi-GPU context works
        by sending the sub-forest from each worker to the client,
        concatenating these into one forest with the full
        `n_estimators` set of trees, and sending this combined forest to
        the workers, which will each infer on their local set of data.
        Within the worker, this uses the cuML Forest Inference Library
        (cuml.fil) for high-throughput prediction.

        This allows inference to scale to large datasets, but the forest
        transmission incurs overheads for very large trees. For inference
        on small datasets, this overhead may dominate prediction time.

        The 'CPU' fallback method works with sub-forests in-place,
        broadcasting the datasets to all workers and combining predictions
        via a voting method at the end. This method is slower
        on a per-row basis but may be faster for problems with many trees
        and few rows.

        In the 0.15 cuML release, inference will be updated with much
        faster tree transfer. Preliminary builds with this updated approach
        will be available from rapids.ai

        Parameters
        ----------
        X : Dask cuDF dataframe  or CuPy backed Dask Array (n_rows, n_features)
            Distributed dense matrix (floats or doubles) of shape
            (n_samples, n_features).
        output_class : boolean (default = True)
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
            `algo` - choose the algorithm automatically. Currently
            'batch_tree_reorg' is used for dense storage
            and 'naive' for sparse storage
        threshold : float (default = 0.5)
            Threshold used for classification. Optional and required only
            while performing the predict operation on the GPU, that is for,
            predict_model='GPU'.
            It is applied if output_class == True, else it is ignored
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
            eagerly executed one.  It is not required  while using
            predict_model='CPU'.

        Returns
        ----------
        y : Dask cuDF dataframe or CuPy backed Dask Array (n_rows, 1)

        """
        if predict_model == "CPU" or self.num_classes > 2:
            preds = self.predict_model_on_cpu(X,
                                              convert_dtype=convert_dtype)

        else:
            preds = \
                self._predict_using_fil(X, output_class=output_class,
                                        algo=algo,
                                        threshold=threshold,
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
    def predict_model_on_cpu(self, X, convert_dtype=True):
        """
        Predicts the labels for X.

        Parameters
        ----------
        X : Dask cuDF dataframe  or CuPy backed Dask Array (n_rows, n_features)
            Distributed dense matrix (floats or doubles) of shape
            (n_samples, n_features).
        convert_dtype : bool, optional (default = True)
            When set to True, the predict method will, when necessary, convert
            the input to the data type which was used to train the model. This
            will increase memory used for the method.
        Returns
        ----------
        y : Dask cuDF dataframe or CuPy backed Dask Array (n_rows, 1)
        """
        c = default_client()
        workers = self.workers

        X_Scattered = c.scatter(X)
        futures = list()
        for n, w in enumerate(workers):
            futures.append(
                c.submit(
                    RandomForestClassifier._predict_model_on_cpu,
                    self.rfs[w],
                    X_Scattered,
                    convert_dtype,
                    workers=[w],
                )
            )

        rslts = self.client.gather(futures, errors="raise")
        indexes = np.zeros(len(futures), dtype=np.int32)
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

    def predict_proba(self, X,
                      delayed=True, **kwargs):
        """
        Predicts the probability of each class for X.

        See documentation of `predict' for notes on performance.

        Parameters
        ----------
        X : Dask cuDF dataframe  or CuPy backed Dask Array (n_rows, n_features)
            Distributed dense matrix (floats or doubles) of shape
            (n_samples, n_features).
        predict_model : String (default = 'GPU')
            'GPU' to predict using the GPU, 'CPU' otherwise. The 'GPU' can only
            be used if the model was trained on float32 data and `X` is float32
            or convert_dtype is set to True. Also the 'GPU' should only be
            used for binary classification problems.
        output_class : boolean (default = True)
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
        threshold : float (default = 0.5)
            Threshold used for classification. Optional and required only
            while performing the predict operation on the GPU.
            It is applied if output_class == True, else it is ignored
        num_classes : int (default = 2)
            number of different classes present in the dataset
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
           Dask cuDF dataframe or CuPy backed Dask Array (n_rows, n_classes)
        """
        if self._get_internal_model() is None:
            self._set_internal_model(self._concat_treelite_models())
        data = DistributedDataHandler.create(X, client=self.client)
        self.datatype = data.datatype
        return self._predict_proba(X, delayed, **kwargs)

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
        params : dict of new params.
        """
        return self._set_params(**params)

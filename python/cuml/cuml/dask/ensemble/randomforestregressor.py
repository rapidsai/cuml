#
# Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
from cuml.dask.common.base import DelayedPredictionMixin
from cuml.ensemble import RandomForestRegressor as cuRFR
from cuml.dask.ensemble.base import BaseRandomForestModel
from cuml.dask.common.base import BaseEstimator

import dask


class RandomForestRegressor(
    BaseRandomForestModel, DelayedPredictionMixin, BaseEstimator
):
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

    The distributed algorithm uses an *embarrassingly-parallel*
    approach. For a forest with `N` trees being built on `w` workers, each
    worker simply builds `N/w` trees on the data it has available
    locally. In many cases, partitioning the data so that each worker
    builds trees on a subset of the total dataset works well, but
    it generally requires the data to be well-shuffled in advance.
    Alternatively, callers can replicate all of the data across
    workers so that ``rf.fit`` receives `w` partitions, each containing the
    same data. This would produce results approximately identical to
    single-GPU fitting.

    Please check the single-GPU implementation of Random Forest
    classifier for more information about the underlying algorithm.

    Parameters
    ----------
    n_estimators : int (default = 100)
        total number of trees in the forest (not per-worker)
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    split_criterion : int or string (default = ``2`` (``'mse'``))
        The criterion used to split nodes.\n
         * ``0`` or ``'gini'`` for gini impurity
         * ``1`` or ``'entropy'`` for information gain (entropy)
         * ``2`` or ``'mse'`` for mean squared error
         * ``4`` or ``'poisson'`` for poisson half deviance
         * ``5`` or ``'gamma'`` for gamma half deviance
         * ``6`` or ``'inverse_gaussian'`` for inverse gaussian deviance

        ``0``, ``'gini'``, ``1``, ``'entropy'`` not valid for regression
    bootstrap : boolean (default = True)
        Control bootstrapping.\n
         * If ``True``, each tree in the forest is built on a bootstrapped
           sample with replacement.
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
        Maximum leaf nodes per tree. Soft constraint. Unlimited, If ``-1``.
    max_features : float (default = 'auto')
        Ratio of number of features (columns) to consider
        per node split.\n
         * If type ``int`` then ``max_features`` is the absolute count of
           features to be used.
         * If type ``float`` then ``max_features`` is a fraction.
         * If ``'auto'`` then ``max_features=n_features = 1.0``.
         * If ``'sqrt'`` then ``max_features=1/sqrt(n_features)``.
         * If ``'log2'`` then ``max_features=log2(n_features)/n_features``.
         * If ``None``, then ``max_features = 1.0``.
    n_bins : int (default = 128)
        Maximum number of bins used by the split algorithm per feature.
    min_samples_leaf : int or float (default = 1)
        The minimum number of samples (rows) in each leaf node.\n
         * If type ``int``, then ``min_samples_leaf`` represents the minimum
           number.
         * If ``float``, then ``min_samples_leaf`` represents a fraction and
           ``ceil(min_samples_leaf * n_rows)`` is the minimum number of
           samples for each leaf node.
    min_samples_split : int or float (default = 2)
        The minimum number of samples required to split an internal node.\n
         * If type ``int``, then ``min_samples_split`` represents the minimum
           number.
         * If type ``float``, then ``min_samples_split`` represents a fraction
           and ``ceil(min_samples_split * n_rows)`` is the minimum number of
           samples for each split.
    accuracy_metric : string (default = 'r2')
        Decides the metric used to evaluate the performance of the model.
        In the 0.16 release, the default scoring metric was changed
        from mean squared error to r-squared.\n
         * for r-squared : ``'r2'``
         * for median of abs error : ``'median_ae'``
         * for mean of abs error : ``'mean_ae'``
         * for mean square error' : ``'mse'``
    n_streams : int (default = 4 )
        Number of parallel streams used for forest building
    workers : optional, list of strings
        Dask addresses of workers to use for computation.
        If None, all available Dask workers will be used.
    random_state : int (default = None)
        Seed for the random number generator. Unseeded by default.

    ignore_empty_partitions: Boolean (default = False)
        Specify behavior when a worker does not hold any data
        while splitting. When True, it returns the results from workers
        with data (the number of trained estimators will be less than
        n_estimators) When False, throws a RuntimeError.
        This is an experimental parameter, and may be removed
        in the future.

    """

    def __init__(
        self,
        *,
        workers=None,
        client=None,
        verbose=False,
        n_estimators=100,
        random_state=None,
        ignore_empty_partitions=False,
        **kwargs,
    ):
        super().__init__(client=client, verbose=verbose, **kwargs)

        self._create_model(
            model_func=RandomForestRegressor._construct_rf,
            client=client,
            workers=workers,
            n_estimators=n_estimators,
            base_seed=random_state,
            ignore_empty_partitions=ignore_empty_partitions,
            **kwargs,
        )

    @staticmethod
    def _construct_rf(n_estimators, random_state, **kwargs):
        return cuRFR(
            n_estimators=n_estimators, random_state=random_state, **kwargs
        )

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

    def fit(self, X, y, convert_dtype=False, broadcast_data=False):
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
                                                           X_dask_cudf:workers,
                                                           y_dask_cudf:workers
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
        broadcast_data : bool, optional (default = False)
            When set to True, the whole dataset is broadcasted
            to train the workers, otherwise each worker
            is trained on its partition

        """
        self.internal_model = None
        self._fit(
            model=self.rfs,
            dataset=(X, y),
            convert_dtype=convert_dtype,
            broadcast_data=broadcast_data,
        )
        return self

    def predict(
        self,
        X,
        predict_model="GPU",
        algo="auto",
        convert_dtype=True,
        fil_sparse_format="auto",
        delayed=True,
        broadcast_data=False,
    ):
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
        X : Dask cuDF dataframe or CuPy backed Dask Array (n_rows, n_features)
            Distributed dense matrix (floats or doubles) of shape
            (n_samples, n_features).
        algo : string (default = 'auto')
            This is optional and required only while performing the
            predict operation on the GPU.

             * ``'naive'`` - simple inference using shared memory
             * ``'tree_reorg'`` - similar to naive but trees rearranged to be
               more coalescing-friendly
             * ``'batch_tree_reorg'`` - similar to tree_reorg but predicting
               multiple rows per thread block
             * ``'auto'`` - choose the algorithm automatically. (Default)
             * ``'batch_tree_reorg'`` is used for dense storage
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

             * ``'auto'`` - choose the storage type automatically
               (currently True is chosen by auto)
             * ``False`` - create a dense forest
             * ``True`` - create a sparse forest, requires algo='naive'
               or algo='auto'

        delayed : bool (default = True)
            Whether to do a lazy prediction (and return Delayed objects) or an
            eagerly executed one.
        broadcast_data : bool (default = False)
            If broadcast_data=False, the trees are merged in a single model
            before the workers perform inference on their share of the
            prediction workload. When broadcast_data=True, trees aren't merged.
            Instead each of the workers infer the whole prediction work
            from trees at disposal. The results are reduced on the client.
            May be advantageous when the model is larger than the data used
            for inference.

        Returns
        -------
        y : Dask cuDF dataframe or CuPy backed Dask Array (n_rows, 1)

        """
        if predict_model == "CPU":
            preds = self.predict_model_on_cpu(X, convert_dtype=convert_dtype)

        else:
            if broadcast_data:
                preds = self.partial_inference(
                    X,
                    algo=algo,
                    convert_dtype=convert_dtype,
                    fil_sparse_format=fil_sparse_format,
                    delayed=delayed,
                )
            else:
                preds = self._predict_using_fil(
                    X,
                    algo=algo,
                    convert_dtype=convert_dtype,
                    fil_sparse_format=fil_sparse_format,
                    delayed=delayed,
                )
        return preds

    def partial_inference(self, X, delayed, **kwargs):
        partial_infs = self._partial_inference(
            X=X, op_type="regression", delayed=delayed, **kwargs
        )

        def reduce(partial_infs, workers_weights, unique_classes=None):
            regressions = dask.array.average(
                partial_infs, axis=1, weights=workers_weights
            )
            merged_regressions = regressions.compute()
            return merged_regressions

        datatype = (
            "daskArray" if isinstance(X, dask.array.Array) else "daskDataframe"
        )

        return self.apply_reduction(reduce, partial_infs, datatype, delayed)

    def predict_using_fil(self, X, delayed, **kwargs):
        if self._get_internal_model() is None:
            self._set_internal_model(self._concat_treelite_models())
        return self._predict_using_fil(X=X, delayed=delayed, **kwargs)

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
        ----------
        deep : boolean (default = True)
        """
        return self._get_params(deep)

    def set_params(self, **params):
        """
        Sets the value of parameters required to
        configure this estimator, it functions similar to
        the sklearn set_params.

        Parameters
        ----------
        params : dict of new params.
        """
        return self._set_params(**params)

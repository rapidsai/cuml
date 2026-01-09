# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#

import cupy as cp
from dask.distributed import get_worker
from raft_dask.common.comms import Comms, get_raft_comm_state

from cuml.dask.common.base import (
    BaseEstimator,
    DelayedPredictionMixin,
    DelayedTransformMixin,
    mnmg_import,
)
from cuml.dask.common.input_utils import DistributedDataHandler, concatenate
from cuml.dask.common.utils import wait_and_raise_from_futures
from cuml.internals.utils import check_random_seed


class KMeans(BaseEstimator, DelayedPredictionMixin, DelayedTransformMixin):
    """
    Multi-Node Multi-GPU implementation of KMeans.

    This version minimizes data transfer by sharing only
    the centroids between workers in each iteration.

    Predictions are done embarrassingly parallel, using cuML's
    single-GPU version.

    For more information on this implementation, refer to the
    documentation for single-GPU K-Means.

    Parameters
    ----------
    handle : cuml.Handle or None, default=None

        .. deprecated:: 26.02
            The `handle` argument was deprecated in 26.02 and will be removed
            in 26.04. There's no need to pass in a handle, cuml now manages
            this resource automatically.

    n_clusters : int (default = 8)
        The number of centroids or clusters you want.
    max_iter : int (default = 300)
        The more iterations of EM, the more accurate, but slower.
    tol : float (default = 1e-4)
        Stopping criterion when centroid means do not change much.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    random_state : int or None (default = None)
        If you want results to be the same when you restart Python,
        select a state.
    init : {'scalable-kmeans++', 'k-means||' , 'random' or an ndarray} \
           (default = 'scalable-k-means++')
        'scalable-k-means++' or 'k-means||': Uses fast and stable scalable
        kmeans++ initialization.
        'random': Choose 'n_cluster' observations (rows) at random
        from data for the initial centroids. If an ndarray is passed,
        it should be of shape (n_clusters, n_features) and gives the
        initial centers.
    oversampling_factor : int (default = 2)
        The amount of points to sample in scalable k-means++ initialization for
        potential centroids. Increasing this value can lead to better initial
        centroids at the cost of memory. The total number of centroids sampled
        in scalable k-means++ is oversampling_factor * n_clusters * 8.
    max_samples_per_batch : int (default = 32768)
        The number of data samples to use for batches of the pairwise distance
        computation. This computation is done throughout both fit predict.
        The default should suit most cases. The total number of elements in the
        batched pairwise distance computation is max_samples_per_batch
        * n_clusters. It might become necessary to lower this number when
        n_clusters becomes prohibitively large.

    Attributes
    ----------
    cluster_centers_ : cuDF DataFrame or CuPy ndarray
        The coordinates of the final clusters. This represents of "mean" of
        each data cluster.

    """

    def __init__(self, *, client=None, verbose=False, **kwargs):
        super().__init__(client=client, verbose=verbose, **kwargs)

    @staticmethod
    @mnmg_import
    def _func_fit(sessionId, objs, datatype, has_weights, **kwargs):
        from cuml.cluster.kmeans_mg import KMeansMG

        handle = get_raft_comm_state(sessionId, get_worker())["handle"]

        if not has_weights:
            inp_data = concatenate(objs)
            inp_weights = None
        else:
            inp_data = concatenate([X for X, weights in objs])
            inp_weights = concatenate([weights for X, weights in objs])

        return KMeansMG(handle=handle, output_type=datatype, **kwargs).fit(
            inp_data, sample_weight=inp_weights
        )

    @staticmethod
    def _score(model, data, sample_weight=None):
        ret = model.score(data, sample_weight=sample_weight)
        return ret

    @staticmethod
    def _check_normalize_sample_weight(sample_weight):
        if sample_weight is not None:
            n_samples = len(sample_weight)
            scale = n_samples / sample_weight.sum()
            sample_weight *= scale
        return sample_weight

    def fit(self, X, sample_weight=None):
        """
        Fit a multi-node multi-GPU KMeans model

        Parameters
        ----------
        X : Dask cuDF DataFrame or CuPy backed Dask Array
        Training data to cluster.

        sample_weight : Dask cuDF DataFrame or CuPy backed Dask Array \
                shape = (n_samples,), default=None # noqa

            The weights for each observation in X. If None, all observations
            are assigned equal weight.
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        """

        sample_weight = self._check_normalize_sample_weight(sample_weight)

        inputs = X if sample_weight is None else (X, sample_weight)

        data = DistributedDataHandler.create(inputs, client=self.client)
        self.datatype = data.datatype

        # Ensure a consistent `random_state` across all calls
        kwargs = self.kwargs.copy()
        kwargs["random_state"] = check_random_seed(kwargs.get("random_state"))

        # This needs to happen on the scheduler
        comms = Comms(comms_p2p=False, client=self.client)
        comms.init(workers=data.workers)

        kmeans_fit = [
            self.client.submit(
                KMeans._func_fit,
                comms.sessionId,
                wf[1],
                self.datatype,
                data.multiple,
                **kwargs,
                workers=[wf[0]],
                pure=False,
            )
            for idx, wf in enumerate(data.worker_to_parts.items())
        ]

        wait_and_raise_from_futures(kmeans_fit)

        comms.destroy()

        models = [res.result() for res in kmeans_fit]
        first = models[0]
        first.labels_ = cp.concatenate([model.labels_ for model in models])
        first.inertia_ = sum(model.inertia_ for model in models)
        self._set_internal_model(first)

        return self

    def fit_predict(self, X, sample_weight=None, delayed=True):
        """
        Compute cluster centers and predict cluster index for each sample.

        Parameters
        ----------
        X : Dask cuDF DataFrame or CuPy backed Dask Array
            Data to predict

        Returns
        -------
        result: Dask cuDF DataFrame or CuPy backed Dask Array
            Distributed object containing predictions

        """
        return self.fit(X, sample_weight=sample_weight).predict(
            X, delayed=delayed
        )

    def predict(self, X, delayed=True):
        """
        Predict labels for the input

        Parameters
        ----------
        X : Dask cuDF DataFrame or CuPy backed Dask Array
            Data to predict

        delayed : bool (default = True)
            Whether to do a lazy prediction (and return Delayed objects) or an
            eagerly executed one.

        Returns
        -------
        result: Dask cuDF DataFrame or CuPy backed Dask Array
            Distributed object containing predictions
        """
        return self._predict(X, delayed=delayed)

    def fit_transform(self, X, sample_weight=None, delayed=True):
        """
        Calls fit followed by transform using a distributed KMeans model

        Parameters
        ----------
        X : Dask cuDF DataFrame or CuPy backed Dask Array
            Data to predict

        delayed : bool (default = True)
            Whether to execute as a delayed task or eager.

        Returns
        -------
        result: Dask cuDF DataFrame or CuPy backed Dask Array
            Distributed object containing the transformed data
        """
        return self.fit(X, sample_weight=sample_weight).transform(
            X, delayed=delayed
        )

    def transform(self, X, delayed=True):
        """
        Transforms the input into the learned centroid space

        Parameters
        ----------
        X : Dask cuDF DataFrame or CuPy backed Dask Array
            Data to predict

        delayed : bool (default = True)
            Whether to execute as a delayed task or eager.

        Returns
        -------
        result: Dask cuDF DataFrame or CuPy backed Dask Array
            Distributed object containing the transformed data
        """
        return self._transform(X, n_dims=2, delayed=delayed)

    def score(self, X, sample_weight=None):
        """
        Computes the inertia score for the trained KMeans centroids.

        Parameters
        ----------
        X : dask_cudf.Dataframe
            Dataframe to compute score

        Returns
        -------

        Inertial score
        """

        sample_weight = self._check_normalize_sample_weight(sample_weight)

        scores = self._run_parallel_func(
            KMeans._score,
            X,
            sample_weight=sample_weight,
            n_dims=1,
            delayed=False,
            output_futures=True,
        )
        return -1.0 * sum(
            -1.0 * score for score in self.client.compute(scores, sync=True)
        )

    def _get_param_names(self):
        return list(self.kwargs.keys())

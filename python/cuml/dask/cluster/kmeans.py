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

from cuml.dask.common import extract_ddf_partitions, to_dask_cudf
from dask.distributed import default_client
from cuml.dask.common.comms import worker_state, CommsContext
from dask.distributed import wait
import numpy as np

import cudf

import random


def concat(dfs):
    if len(dfs) == 1:
        return dfs[0]
    else:
        return cudf.concat(dfs)


class KMeans(object):
    """
    Multi-Node Multi-GPU implementation of KMeans.

    This version minimizes data transfer by sharing only
    the centroids between workers in each iteration.

    Predictions are done embarrassingly parallel, using cuML's
    single-GPU version.

    For more information on this implementation, refer to the
    documentation for single-GPU K-Means.
    """

    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4,
                 verbose=0, random_state=1, precompute_distances='auto',
                 init='scalable-k-means++', n_init=1, algorithm='auto',
                 client=None):
        """
        Constructor for distributed KMeans model
        handle : cuml.Handle
            If it is None, a new one is created just for this class.
        n_clusters : int (default = 8)
            The number of centroids or clusters you want.
        max_iter : int (default = 300)
            The more iterations of EM, the more accurate, but slower.
        tol : float (default = 1e-4)
            Stopping criterion when centroid means do not change much.
        verbose : boolean (default = 0)
            If True, prints diagnositc information.
        random_state : int (default = 1)
            If you want results to be the same when you restart Python,
            select a state.
        precompute_distances : boolean (default = 'auto')
            Not supported yet.
        init : {'scalable-kmeans++', 'k-means||' , 'random' or an ndarray}
               (default = 'scalable-k-means++')
            'scalable-k-means++' or 'k-means||': Uses fast and stable scalable
            kmeans++ intialization.
            'random': Choose 'n_cluster' observations (rows) at random
            from data for the initial centroids. If an ndarray is passed,
            it should be of shape (n_clusters, n_features) and gives the
            initial centers.
        n_init : int (default = 1)
            Number of times intialization is run. More is slower,
            but can be better.
        algorithm : "auto"
            Currently uses full EM, but will support others later.
        n_gpu : int (default = 1)
            Number of GPUs to use. Currently uses single GPU, but will support
            multiple GPUs later.
        """
        self.client = default_client() if client is None else client
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.precompute_distances = precompute_distances
        self.n_init = n_init
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        self.init = init
        self.verbose = verbose

    @staticmethod
    def func_fit(sessionId, n_clusters, max_iter, tol, verbose, random_state,
                 precompute_distances, init, n_init, algorithm, dfs, r):
        """
        Runs on each worker to call fit on local KMeans instance.
        Extracts centroids
        :param model: Local KMeans instance
        :param dfs: List of cudf.Dataframes to use
        :param r: Stops memoizatiion caching
        :return: The fit model
        """
        from cuml.cluster.kmeans_mg import KMeansMG as cumlKMeans
        handle = worker_state(sessionId)["handle"]

        df = concat(dfs)

        return cumlKMeans(handle=handle,
                          init=init,
                          max_iter=max_iter,
                          tol=tol,
                          random_state=random_state,
                          n_init=n_init,
                          algorithm=algorithm,
                          precompute_distances=precompute_distances,
                          n_clusters=n_clusters,
                          verbose=verbose).fit(df)

    @staticmethod
    def func_transform(model, dfs, r):
        """
        Runs on each worker to call fit on local KMeans instance
        :param model: Local KMeans instance
        :param dfs: List of cudf.Dataframes to use
        :param r: Stops memoizatiion caching
        :return: The fit model
        """

        df = concat(dfs)
        return model.transform(df)

    @staticmethod
    def func_predict(model, dfs, r):
        """
        Runs on each worker to call fit on local KMeans instance
        :param model: Local KMeans instance
        :param dfs: List of cudf.Dataframes to use
        :param r: Stops memoization caching
        :return: cudf.Series with predictions
        """
        df = concat(dfs)
        return model.predict(df)

    @staticmethod
    def func_score(model, dfs, r):
        """
        Runs on each worker to call fit on local KMeans instance
        :param model: Local KMeans instance
        :param dfs: List of cudf.Dataframes to use
        :param r: Stops memoization caching
        :return: cudf.Series with predictions
        """
        df = concat(dfs)
        return model.score(df)

    def fit(self, X):
        """
        Fits a distributed KMeans model
        :param X: dask_cudf.Dataframe to fit
        :return: This KMeans instance
        """
        gpu_futures = self.client.sync(extract_ddf_partitions, X)

        workers = list(map(lambda x: x[0], gpu_futures.items()))

        comms = CommsContext(comms_p2p=False)
        comms.init(workers=workers)

        kmeans_fit = [self.client.submit(
            KMeans.func_fit,
            comms.sessionId,
            self.n_clusters,
            self.max_iter,
            self.tol,
            self.verbose,
            self.random_state,
            self.precompute_distances,
            self.init,
            self.n_init,
            self.algorithm,
            f,
            random.random(),
            workers=[w]) for w, f in gpu_futures.items()]

        wait(kmeans_fit)

        comms.destroy()

        self.local_model = kmeans_fit[0].result()
        self.cluster_centers_ = self.local_model.cluster_centers_

        return self

    def parallel_func(self, X, func):
        """
        Predicts the labels using a distributed KMeans model
        :param X: dask_cudf.Dataframe to predict
        :return: A dask_cudf.Dataframe containing label predictions
        """
        gpu_futures = self.client.sync(extract_ddf_partitions, X)
        kmeans_predict = [self.client.submit(
            func,
            self.local_model,
            f,
            random.random(),
            workers=[w]) for w, f in gpu_futures.items()]

        return to_dask_cudf(kmeans_predict)

    def predict(self, X):
        """
        Predicts the labels using a distributed KMeans model
        :param X: dask_cudf.Dataframe to predict
        :return: A dask_cudf.Dataframe containing label predictions
        """
        return self.parallel_func(X, KMeans.func_predict)

    def fit_predict(self, X):
        return self.fit(X).predict(X)

    def transform(self, X):
        """
        Predicts the labels using a distributed KMeans model
        :param X: dask_cudf.Dataframe to predict
        :return: A dask_cudf.Dataframe containing label predictions
        """
        return self.parallel_func(X, KMeans.func_xform)

    def fit_transform(self, X):
        """
        Calls fit followed by transform using a distributed KMeans model
        :param X: dask_cudf.Dataframe to fit & predict
        :return: A dask_cudf.Dataframe containing label predictions
        """
        return self.fit(X).transform(X)

    def score(self, X):
        gpu_futures = self.client.sync(extract_ddf_partitions, X)
        scores = [self.client.submit(
            KMeans.func_score,
            self.local_model,
            f,
            random.random(),
            workers=[w]).result() for w, f in gpu_futures.items()]

        return np.sum(scores)

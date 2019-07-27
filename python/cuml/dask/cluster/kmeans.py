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

import random


class KMeans(object):
    """
    Multi-Node Multi-GPU implementation of KMeans
    """

    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4,
                 verbose=0, random_state=1, precompute_distances='auto',
                 init='scalable-k-means++', n_init=1, algorithm='auto',
                 client=None):
        """
        Constructor for distributed KMeans model
        :param n_clusters: Number of clusters to fit
        :param init_method: Method for finding initial centroids
        :param verbose: Print useful info while executing
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
                 precompute_distances, init, n_init, algorithm, df, r):
        """
        Runs on each worker to call fit on local KMeans instance.
        Extracts centroids
        :param model: Local KMeans instance
        :param df: cudf.Dataframe to use
        :param r: Stops memoizatiion caching
        :return: The fit model
        """
        from cuml.cluster.kmeans_mg import KMeansMG as cumlKMeans
        handle = worker_state(sessionId)["handle"]
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
    def func_transform(model, df, r):
        """
        Runs on each worker to call fit on local KMeans instance
        :param model: Local KMeans instance
        :param df: cudf.Dataframe to use
        :param r: Stops memoizatiion caching
        :return: The fit model
        """
        return model.transform(df)

    @staticmethod
    def func_predict(model, df, r):
        """
        Runs on each worker to call fit on local KMeans instance
        :param model: Local KMeans instance
        :param df: cudf.Dataframe to use
        :param r: Stops memoization caching
        :return: cudf.Series with predictions
        """
        return model.predict(df)

    @staticmethod
    def func_score(model, df, r):
        """
        Runs on each worker to call fit on local KMeans instance
        :param model: Local KMeans instance
        :param df: cudf.Dataframe to use
        :param r: Stops memoization caching
        :return: cudf.Series with predictions
        """
        return model.score(df)

    def fit(self, X):
        """
        Fits a distributed KMeans model
        :param X: dask_cudf.Dataframe to fit
        :return: This KMeans instance
        """
        gpu_futures = self.client.sync(extract_ddf_partitions, X)

        workers = list(map(lambda x: x[0], gpu_futures))

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
            workers=[w]) for w, f in gpu_futures]

        wait(kmeans_fit)

        comms.destroy()

        self.local_model = kmeans_fit[0].result()

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
            workers=[w]) for w, f in gpu_futures]

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
            workers=[w]).result() for w, f in gpu_futures]

        return np.sum(scores)

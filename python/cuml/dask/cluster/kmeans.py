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
from cuml.cluster import KMeans as cumlKMeans
from cuml.dask.common.comms import worker_state, CommsContext
from dask.distributed import wait

import random


class KMeans(object):
    """
    Multi-Node Multi-GPU implementation of KMeans
    """

    def __init__(self, n_clusters=8, init="k-means||", verbose=0):
        """
        Constructor for distributed KMeans model
        :param n_clusters: Number of clusters to fit
        :param init_method: Method for finding initial centroids
        :param verbose: Print useful info while executing
        """
        self.init(n_clusters=n_clusters, init=init,
                  verbose=verbose)

    def init(self, n_clusters, init, verbose=0):
        """
        Creates a local KMeans instance on each worker
        :param n_clusters: Number of clusters to fit
        :param init_method: Method for finding initial centroids
        :param verbose: Print useful info while executing
        :return:
        """
        self.comms = CommsContext(comms_p2p=False)
        self.comms.init(list(self.comms.client.has_what().keys()))

        self.kmeans = [(w, self.comms.client.submit(KMeans.func_build_kmeans_,
                                                    self.comms.sessionId,
                                                    n_clusters,
                                                    init,
                                                    verbose,
                                                    i,
                                                    workers=[w]))
                       for i, w in zip(range(len(self.comms.worker_addresses)),
                                       self.comms.workers)]
        wait(self.kmeans)

    @staticmethod
    def func_build_kmeans_(sessionId, n_clusters, init, verbose, r):
        """
        Create local KMeans instance on worker
        :param handle: instance of cuml.handle.Handle
        :param n_clusters: Number of clusters to fit
        :param init_method: Method for finding initial centroids
        :param verbose: Print useful info while executing
        :param r: Stops memoization caching
        """
        handle = worker_state(sessionId)["handle"]
        return cumlKMeans(handle=handle, init=init,
                          n_clusters=n_clusters, verbose=verbose)

    @staticmethod
    def func_fit(model, df, r):
        """
        Runs on each worker to call fit on local KMeans instance
        :param model: Local KMeans instance
        :param df: cudf.Dataframe to use
        :param r: Stops memoizatiion caching
        :return: The fit model
        """
        return model.fit(df)

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

    def run_model_func_on_dask_cudf(self, func, X):
        """
        Runs a function on a local KMeans instance on each worker
        :param func: The function to execute on each worker
        :param X: Input dask_cudf.Dataframe
        :return: Futures containing results of func
        """
        gpu_futures = self.comms.client.sync(extract_ddf_partitions, X)

        worker_model_map = dict(map(lambda x: (x[0], x[1]), self.kmeans))

        f = [self.comms.client.submit(func,  # Function to run on worker
                                      worker_model_map[w],  # Model instance
                                      f,  # Input DataFrame partition
                                      random.random())  # Worker ID
             for w, f in gpu_futures]
        wait(f)
        return f

    def fit(self, X):
        """
        Fits a distributed KMeans model
        :param X: dask_cudf.Dataframe to fit
        :return: This KMeans instance
        """
        self.run_model_func_on_dask_cudf(KMeans.func_fit, X)

        # TODO: After fitting model, pull the centroids, inertia, & n_iters to client.

        return self

    def predict(self, X):
        """
        Predicts the labels using a distributed KMeans model
        :param X: dask_cudf.Dataframe to predict
        :return: A dask_cudf.Dataframe containing label predictions
        """

        # TODO: Propagate centroids, inertia, and n_iters to workers and
        # Run regular predict in parallel.

        f = self.run_model_func_on_dask_cudf(KMeans.func_predict, X)
        return to_dask_cudf(f)

    def transform(self, X):
        """
        Transform X to a cluster-distance space.

        Parameters
        ----------
        X : dask_cudf.Dataframe shape = (n_samples, n_features)
        """

        # TODO: Propagate centroids, inertia, and n_iters to workers and
        # run regular transform.

        f = self.run_model_func_on_dask_cudf(KMeans.func_transform, X)
        return to_dask_cudf(f)

    def fit_transform(self, X):
        """
        Compute clustering and transform X to cluster-distance space.

        Parameters
        ----------
        X : dask_cudf.Dataframe shape = (n_samples, n_features)
        """
        return self.fit(X).transform(X)

    def fit_predict(self, X):
        """
        Calls fit followed by predict using a distributed KMeans model
        :param X: dask_cudf.Dataframe to fit & predict
        :return: A dask_cudf.Dataframe containing label predictions
        """
        return self.fit(X).predict(X)

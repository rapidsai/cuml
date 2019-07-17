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
from cuml.dask.common.comms import worker_state, default_comms
from dask.distributed import wait

import random


class KMeans(object):
    """
    Multi-Node Multi-GPU implementation of KMeans
    """

    def __init__(self, n_clusters=8, init_method="random", verbose=0):
        """
        Constructor for distributed KMeans model
        :param n_clusters: Number of clusters to fit
        :param init_method: Method for finding initial centroids
        :param verbose: Print useful info while executing
        """
        self.init(n_clusters=n_clusters, init_method=init_method,
                  verbose=verbose)

    def init(self, n_clusters, init_method, verbose=0):
        """
        Creates a local KMeans instance on each worker
        :param n_clusters: Number of clusters to fit
        :param init_method: Method for finding initial centroids
        :param verbose: Print useful info while executing
        :return:
        """

        comms = default_comms()

        self.kmeans = [(w, comms.client.submit(KMeans.func_build_kmeans_,
                                               comms.sessionId,
                                               n_clusters,
                                               init_method,
                                               verbose,
                                               i,
                                               workers=[w]))
                       for i, w in zip(range(len(comms.worker_addresses)),
                                       comms.workers)]
        wait(self.kmeans)

    @staticmethod
    def func_build_kmeans_(sessionId, n_clusters, init_method, verbose, r):
        """
        Create local KMeans instance on worker
        :param handle: instance of cuml.handle.Handle
        :param n_clusters: Number of clusters to fit
        :param init_method: Method for finding initial centroids
        :param verbose: Print useful info while executing
        :param r: Stops memoization caching
        """
        handle = worker_state(sessionId)["handle"]
        return cumlKMeans(handle=handle, init=init_method,
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
        comms = default_comms()

        gpu_futures = comms.client.sync(extract_ddf_partitions, X)

        worker_model_map = dict(map(lambda x: (x[0], x[1]), self.kmeans))

        f = [comms.client.submit(func,  # Function to run on worker
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
        return self

    def predict(self, X):
        """
        Predicts the labels using a distributed KMeans model
        :param X: dask_cudf.Dataframe to predict
        :return: A dask_cudf.Dataframe containing label predictions
        """
        f = self.run_model_func_on_dask_cudf(KMeans.func_predict, X)
        return to_dask_cudf(f)

    def fit_predict(self, X):
        """
        Calls fit followed by predict using a distributed KMeans model
        :param X: dask_cudf.Dataframe to fit & predict
        :return: A dask_cudf.Dataframe containing label predictions
        """
        return self.fit(X).predict(X)

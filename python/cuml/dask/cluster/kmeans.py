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

    def __init__(self, client=None, **kwargs):
        """
        Constructor for distributed KMeans model

        Parameters
        ----------
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
        init : {'scalable-kmeans++', 'k-means||' , 'random' or an ndarray}
               (default = 'scalable-k-means++')
            'scalable-k-means++' or 'k-means||': Uses fast and stable scalable
            kmeans++ intialization.
            'random': Choose 'n_cluster' observations (rows) at random
            from data for the initial centroids. If an ndarray is passed,
            it should be of shape (n_clusters, n_features) and gives the
            initial centers.
        oversampling_factor : int (default = 2) The amount of points to sample
            in scalable k-means++ initialization for potential centroids.
            Increasing this value can lead to better initial centroids at the
            cost of memory. The total number of centroids sampled in scalable
            k-means++ is oversampling_factor * n_clusters * 8.
        max_samples_per_batch : int (default = 32768) The number of data
            samples to use for batches of the pairwise distance computation.
            This computation is done throughout both fit predict. The default
            should suit most cases. The total number of elements in the
            batched pairwise distance computation is max_samples_per_batch
            * n_clusters. It might become necessary to lower this number when
            n_clusters becomes prohibitively large.

        Attributes
        ----------
        cluster_centers_ : array
            The coordinates of the final clusters. This represents of "mean" of
            each data cluster.
        """
        self.client = default_client() if client is None else client
        self.kwargs = kwargs

    @staticmethod
    def func_fit(sessionId, dfs, r, **kwargs):
        """
        Runs on each worker to call fit on local KMeans instance.
        Extracts centroids
        :param model: Local KMeans instance
        :param dfs: List of cudf.Dataframes to use
        :param r: Stops memoizatiion caching
        :return: The fit model
        """

        try:
            from cuml.cluster.kmeans_mg import KMeansMG as cumlKMeans
        except ImportError:
            raise Exception("cuML has not been built with multiGPU support "
                            "enabled. Build with the --multigpu flag to"
                            " enable multiGPU support.")

        handle = worker_state(sessionId)["handle"]

        df = concat(dfs)

        return cumlKMeans(handle=handle, **kwargs).fit(df)

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
            f,
            random.random(),
            **self.kwargs,
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

    def get_params(self):
        """
        Scikit-learn style return parameter state
        """
        return self.kwargs

    def set_params(self, **params):
        """
        Scikit-learn style set parameter state to dictionary of params.

        Parameters
        -----------
        params : dict of new params
        """
        if not params:
            return self
        current_params = self.kwargs
        for key, value in params.items():
            if key not in current_params:
                raise ValueError('Invalid parameter for estimator')
            else:
                self.kwargs[key] = value
        return self

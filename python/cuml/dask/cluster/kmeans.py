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


import cupy as cp

from cuml.dask.common.input_utils import concatenate
from cuml.dask.common.input_utils import MGData

from cuml.dask.common import raise_mg_import_exception
from dask.distributed import default_client
from cuml.dask.common.comms import worker_state, CommsContext
from dask.distributed import wait

from cuml.dask.common.base import DelayedPredictionMixin
from cuml.dask.common.base import DelayedTransformMixin

from cuml.dask.common.utils import raise_exception_from_futures

from uuid import uuid1


class KMeans(DelayedPredictionMixin, DelayedTransformMixin):
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
    def _func_fit(sessionId, objs, datatype, **kwargs):
        """
        Runs on each worker to call fit on local KMeans instance.
        Extracts centroids
        :param model: Local KMeans instance
        :param dfs: List of cudf.Dataframes to use
        :param r: Stops memoization caching
        :return: The fit model
        """

        try:
            from cuml.cluster.kmeans_mg import KMeansMG as cumlKMeans
        except ImportError:
            raise_mg_import_exception()

        handle = worker_state(sessionId)["handle"]

        inp_data = concatenate(objs)

        return cumlKMeans(handle=handle, output_type=datatype,
                          **kwargs).fit(inp_data)

    @staticmethod
    def _score(model, lock, data):
        lock.acquire()
        ret = model.score(data)
        lock.release()
        return ret

    def fit(self, X):
        """
        Fit a multi-node multi-GPU KMeans model

        Parameters
        ----------
        X : dask_cudf.Dataframe

        Returns
        -------
        self: KMeans model
        """

        data = MGData.single(X, client=self.client)
        self.datatype = data.datatype

        comms = CommsContext(comms_p2p=False, verbose=True)
        comms.init(workers=data.workers)

        key = uuid1()
        kmeans_fit = [self.client.submit(KMeans._func_fit,
                                         comms.sessionId,
                                         wf[1],
                                         self.datatype,
                                         **self.kwargs,
                                         workers=[wf[0]],
                                         key="%s-%s" % (key, idx))
                      for idx, wf in enumerate(data.worker_to_parts.items())]

        wait(kmeans_fit)
        raise_exception_from_futures(kmeans_fit)

        comms.destroy()

        self.local_model = kmeans_fit[0].result()
        self.cluster_centers_ = self.local_model.cluster_centers_

        return self

    def fit_predict(self, X, delayed=True, parallelism=5):
        """
        Compute cluster centers and predict cluster index for each sample.

        Parameters
        ----------
        X : dask_cudf.Dataframe
            Dataframe to predict

        Returns
        -------
        result: dask_cudf.Dataframe
            Dataframe containing predictions

        """
        return self.fit(X).predict(X, delayed, parallelism)

    def predict(self, X, delayed=True, parallelism=5):
        """
        Predict labels for the input

        Parameters
        ----------
        X : dask_cudf.Dataframe
            Dataframe to transform

        delayed : bool delay execution?

        parallelism : integer number of concurrently executing partitions
                              per worker

        Returns
        -------

        result: dask_cudf.Dataframe
            Dataframe containing transform
        """
        return self._predict(X, delayed, parallelism)

    def fit_transform(self, X, delayed=True, parallelism=5):
        """
        Calls fit followed by transform using a distributed KMeans model

        Parameters
        ----------
        X : dask_cudf.Dataframe
            Dataframe to predict

        Returns
        -------
        result: dask_cudf.Dataframe
            Dataframe containing predictions
        """
        return self.fit(X).transform(X, delayed, parallelism)

    def transform(self, X, delayed=True, parallelism=5):
        """
        Transforms the input into the learned centroid space

        Parameters
        ----------
        X : dask_cudf.Dataframe
            Dataframe to transform

        delayed : bool delay execution?

        parallelism : integer number of concurrently executing partitions
                              per worker

        Returns
        -------

        result: dask_cudf.Dataframe
            Dataframe containing transform
        """
        return self._transform(X, delayed, parallelism)

    def score(self, X, parallelism=5):
        """
        Computes the inertia score for the trained KMeans centroids.

        Parameters
        ----------
        X : dask_cudf.Dataframe
            Dataframe to compute score

        parallelism : integer number of concurrently executing partitions
                              per worker.

        Returns
        -------

        Inertial score
        """

        scores = self._run_parallel_func(KMeans._score, X, 1, False, parallelism,
                                         output_futures=True)

        return -1 * cp.sum(cp.asarray(
            self.client.compute(scores, sync=True))*-1.0)

    def get_param_names(self):
        return list(self.kwargs.keys())

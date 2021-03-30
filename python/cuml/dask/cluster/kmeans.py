# Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

from cuml.dask.common.base import BaseEstimator
from cuml.dask.common.base import DelayedPredictionMixin
from cuml.dask.common.base import DelayedTransformMixin
from cuml.dask.common.base import mnmg_import

from cuml.dask.common.input_utils import concatenate
from cuml.dask.common.input_utils import DistributedDataHandler

from cuml.raft.dask.common.comms import Comms
from cuml.raft.dask.common.comms import get_raft_comm_state

from cuml.dask.common.utils import wait_and_raise_from_futures

from cuml.common.memory_utils import with_cupy_rmm


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

    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    n_clusters : int (default = 8)
        The number of centroids or clusters you want.
    max_iter : int (default = 300)
        The more iterations of EM, the more accurate, but slower.
    tol : float (default = 1e-4)
        Stopping criterion when centroid means do not change much.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    random_state : int (default = 1)
        If you want results to be the same when you restart Python,
        select a state.
    init : {'scalable-kmeans++', 'k-means||' , 'random' or an ndarray} \
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

    cluster_centers_ : cuDF DataFrame or CuPy ndarray
        The coordinates of the final clusters. This represents of "mean" of
        each data cluster.

    """

    def __init__(self, *, client=None, verbose=False, **kwargs):
        super().__init__(client=client,
                         verbose=verbose,
                         **kwargs)

    @staticmethod
    @mnmg_import
    def _func_fit(sessionId, objs, datatype, **kwargs):
        from cuml.cluster.kmeans_mg import KMeansMG as cumlKMeans
        handle = get_raft_comm_state(sessionId)["handle"]

        inp_data = concatenate(objs)

        return cumlKMeans(handle=handle, output_type=datatype,
                          **kwargs).fit(inp_data)

    @staticmethod
    def _score(model, data):
        ret = model.score(data)
        return ret

    @with_cupy_rmm
    def fit(self, X):
        """
        Fit a multi-node multi-GPU KMeans model

        Parameters
        ----------
        X : Dask cuDF DataFrame or CuPy backed Dask Array
        Training data to cluster.

        """

        data = DistributedDataHandler.create(X, client=self.client)
        self.datatype = data.datatype

        # This needs to happen on the scheduler
        comms = Comms(comms_p2p=False, client=self.client)
        comms.init(workers=data.workers)

        kmeans_fit = [self.client.submit(KMeans._func_fit,
                                         comms.sessionId,
                                         wf[1],
                                         self.datatype,
                                         **self.kwargs,
                                         workers=[wf[0]],
                                         pure=False)
                      for idx, wf in enumerate(data.worker_to_parts.items())]

        wait_and_raise_from_futures(kmeans_fit)

        comms.destroy()

        self._set_internal_model(kmeans_fit[0])

        return self

    def fit_predict(self, X, delayed=True):
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
        return self.fit(X).predict(X, delayed=delayed)

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

    def fit_transform(self, X, delayed=True):
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
        return self.fit(X).transform(X, delayed=delayed)

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

    @with_cupy_rmm
    def score(self, X):
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

        scores = self._run_parallel_func(KMeans._score,
                                         X,
                                         n_dims=1,
                                         delayed=False,
                                         output_futures=True)

        return -1 * cp.sum(cp.asarray(
            self.client.compute(scores, sync=True))*-1.0)

    def get_param_names(self):
        return list(self.kwargs.keys())

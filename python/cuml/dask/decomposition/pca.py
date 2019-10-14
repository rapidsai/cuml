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

from uuid import uuid1

from functools import reduce

from collections import OrderedDict


class PCA(object):
    """
    Multi-Node Multi-GPU implementation of PCA.

    Predictions are done embarrassingly parallel, using cuML's
    single-GPU version.

    For more information on this implementation, refer to the
    documentation for single-GPU PCA.
    """

    def __init__(self, client=None, **kwargs):
        """
        Constructor for distributed PCA model
        """
        self.client = default_client() if client is None else client
        self.kwargs = kwargs

        # define attributes to make sure they
        # are available even on untrained object
        self.local_model = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.noise_variance = None


    @staticmethod
    def _func_create_model(sessionId, dfs, **kwargs):
        """
        Builds a PCA model on a worker
        :param sessionId:
        :param kwargs:
        :return:
        """
        try:
            from cuml.decomposition.pca_mg import PCAMG as cumlPCA
        except ImportError:
            raise Exception("cuML has not been built with multiGPU support "
                            "enabled. Build with the --multigpu flag to"
                            " enable multiGPU support.")

        handle = worker_state(sessionId)["handle"]
        return cumlPCA(handle=handle, **kwargs), dfs

    @staticmethod
    def _func_fit(f, M, N, partsToRanks, transform):
        """
        Runs on each worker to call fit on local PCA instance.
        Extracts centroids
        :param model: Local KMeans instance
        :param dfs: List of cudf.Dataframes to use
        :param r: Stops memoizatiion caching
        :return: The fit model
        """
        m, dfs = f
        return m.fit(dfs, M, N, partsToRanks, transform)

    @staticmethod
    def _func_get_first(f):
        return f[0]

    @staticmethod
    def _func_xform(model, df):
        """
        Runs on each worker to call fit on local KMeans instance
        :param model: Local KMeans instance
        :param dfs: List of cudf.Dataframes to use
        :param r: Stops memoizatiion caching
        :return: The fit model
        """
        return model.transform(df)

    @staticmethod
    def _func_get_size(df):
        return df.shape[0]

    def fit(self, X, _transform=False):
        """
        Fits a distributed KMeans model
        :param X: dask_cudf.Dataframe to fit
        :return: This KMeans instance
        """
        gpu_futures = self.client.sync(extract_ddf_partitions, X, agg=False)

        worker_to_parts = OrderedDict()
        for w, p in gpu_futures:
            if w not in worker_to_parts:
                worker_to_parts[w] = []
            worker_to_parts[w].append(p)

        workers = list(map(lambda x: x[0], gpu_futures))

        comms = CommsContext(comms_p2p=False)
        comms.init(workers=workers)

        worker_info = comms.worker_info(comms.worker_addresses)

        key = uuid1()
        partsToRanks = [(worker_info[wf[0]]["r"], self.client.submit(
            PCA._func_get_size,
            wf[1],
            workers=[wf[0]],
            key="%s-%s" % (key, idx)).result())
            for idx, wf in enumerate(gpu_futures)]

        N = X.shape[1]
        M = reduce(lambda a,b: a+b, map(lambda x: x[1], partsToRanks))

        key = uuid1()
        pca_models = [(wf[0], self.client.submit(
            PCA._func_create_model,
            comms.sessionId,
            wf[1],
            **self.kwargs,
            workers=[wf[0]],
            key="%s-%s" % (key, idx)))
            for idx, wf in enumerate(worker_to_parts.items())]

        key = uuid1()
        pca_fit = [self.client.submit(
            PCA._func_fit,
            wf[1],
            M, N,
            partsToRanks,
            _transform,
            key="%s-%s" % (key, idx),
            workers=[wf[0]])
            for idx, wf in enumerate(pca_models)]

        wait(pca_fit)

        comms.destroy()

        self.local_model = self.client.submit(PCA._func_get_first,
                                              pca_models[0][1]).result()

        self.components_ = self.local_model.components_
        self.explained_variance_ = self.local_model.explained_variance_
        self.explained_variance_ratio_ = self.local_model.explained_variance_ratio_
        self.singular_values_ = self.local_model.singular_values_
        self.noise_variance = self.local_model.noise_variance_

        if _transform:
            return to_dask_cudf(pca_fit)
        else:
            return self

    def _parallel_func(self, X, func):
        """
        Predicts the labels using a distributed KMeans model
        :param X: dask_cudf.Dataframe to predict
        :return: A dask_cudf.Dataframe containing label predictions
        """
        key = uuid1()
        gpu_futures = self.client.sync(extract_ddf_partitions, X)
        pca_predict = [self.client.submit(
            func,
            self.local_model,
            wf[1],
            workers=[wf[0]],
            key="%s-%s" % (key, idx))
            for idx, wf in enumerate(gpu_futures.items())]

        return to_dask_cudf(pca_predict)

    def transform(self, X):
        """
        Predicts the labels using a distributed KMeans model
        :param X: dask_cudf.Dataframe to predict
        :return: A dask_cudf.Dataframe containing label predictions
        """
        return self.parallel_func(X, PCA._func_xform)

    def fit_transform(self, X):
        """
        Calls fit followed by transform using a distributed KMeans model
        :param X: dask_cudf.Dataframe to fit & predict
        :return: A dask_cudf.Dataframe containing label predictions
        """
        return self.fit(X, _transform=True)

    def get_param_names(self):
        return list(self.kwargs.keys())

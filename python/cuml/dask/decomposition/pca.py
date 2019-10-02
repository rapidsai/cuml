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

    @staticmethod
    def func_fit(sessionId, dfs, M, N, partsToRanks, **kwargs):
        """
        Runs on each worker to call fit on local KMeans instance.
        Extracts centroids
        :param model: Local KMeans instance
        :param dfs: List of cudf.Dataframes to use
        :param r: Stops memoizatiion caching
        :return: The fit model
        """
        try:
            from cuml.decomposition.pca_mg import PCAMG as cumlPCA
        except ImportError:
            raise Exception("cuML has not been built with multiGPU support "
                            "enabled. Build with the --multigpu flag to"
                            " enable multiGPU support.")

        handle = worker_state(sessionId)["handle"]

        return cumlPCA(handle=handle, **kwargs).fit(dfs, M, N, partsToRanks)

    @staticmethod
    def func_xform(model, df):
        """
        Runs on each worker to call fit on local KMeans instance
        :param model: Local KMeans instance
        :param dfs: List of cudf.Dataframes to use
        :param r: Stops memoizatiion caching
        :return: The fit model
        """
        return model.transform(df)

    @staticmethod
    def func_get_size(df):
        return df.shape[0]

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

        # TODO: Build partsToRanks using gpu_futures

        M, N = X.shape
        worker_info = comms.worker_info(workers)

        key = uuid1()
        partsToRanks = [(worker_info[wf[0]]["r"], self.client.submit(
            PCA.func_get_size,
            wf[1],
            wotkers=[wf[0]],
            key="%s-%s" % (key, idx)))
            for idx, wf in enumerate(gpu_futures.items())]

        key = uuid1()
        pca_fit = [self.client.submit(
            PCA.func_fit,
            comms.sessionId,
            wf[1],
            M, N,
            partsToRanks,
            **self.kwargs,
            workers=[wf[0]],
            key="%s-%s" % (key, idx))
            for idx, wf in enumerate(gpu_futures.items())]

        wait(pca_fit)

        comms.destroy()

        self.local_model = pca_fit[0].result()
        self.components_ = self.local_model.components_
        self.explained_variance_ = self.local_model.explained_variance_
        self.explained_variance_ratio_ = self.local_model.explained_variance_ratio_
        self.singular_values_ = self.local_model.singular_values_
        self.noise_variance = self.local_model.noise_variance_

        return self

    def parallel_func(self, X, func):
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
        return self.parallel_func(X, PCA.func_xform)

    def fit_transform(self, X):
        """
        Calls fit followed by transform using a distributed KMeans model
        :param X: dask_cudf.Dataframe to fit & predict
        :return: A dask_cudf.Dataframe containing label predictions
        """
        return self.fit(X).transform(X)

    def get_param_names(self):
        return list(self.kwargs.keys())

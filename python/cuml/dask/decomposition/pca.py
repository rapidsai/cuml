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

from cuml.dask.common import extract_ddf_partitions, to_dask_cudf, raise_exception_from_futures
from dask.distributed import default_client
from cuml.dask.common.comms import worker_state, CommsContext
from dask.distributed import wait

from uuid import uuid1

from functools import reduce

from collections import OrderedDict


class PCA(object):
    """
    PCA (Principal Component Analysis) is a fundamental dimensionality
    reduction technique used to combine features in X in linear combinations
    such that each new component captures the most information or variance of
    the data. N_components is usually small, say at 3, where it can be used for
    data visualization, data compression and exploratory analysis.

    cuML's multi-node multi-gpu (MNMG) PCA expects a dask cuDF input, and provides
    a Full algorithm. Full uses a full eigendecomposition
    then selects the top K eigenvectors. The Jacobi algorithm is much faster
    as it iteratively tries to correct the top K eigenvectors, but might be
    less accurate.

    Examples
    ---------

    .. code-block:: python

        from dask_cuda import LocalCUDACluster
        from dask.distributed import Client, wait
        import numpy as np
        from cuml.dask.decomposition import PCA
        from cuml.dask.datasets import make_blobs

        cluster = LocalCUDACluster(threads_per_worker=1)
        client = Client(cluster)

        nrows = 6
        ncols = 3
        n_parts = 2

        X_cudf, _ = make_blobs(nrows, ncols, 1, n_parts,
                        cluster_std=0.01, verbose=False,
                        random_state=10, dtype=np.float32)

        wait(X_cudf)

        print("Input Matrix")
        print(X_cudf.compute())

        cumlModel = PCA(n_components = 1, whiten=False)
        XT = cumlModel.fit_transform(X_cudf)
        
        print("Transformed Input Matrix")
        print(XT.compute())

    Output:

    .. code-block:: python

          Input Matrix:
                    0         1         2
                    0 -6.520953  0.015584 -8.828546
                    1 -6.507554  0.016524 -8.836799
                    2 -6.518214  0.010457 -8.821301
                    0 -6.520953  0.015584 -8.828546
                    1 -6.507554  0.016524 -8.836799
                    2 -6.518214  0.010457 -8.821301

          Transformed Input Matrix:
                              0
                    0 -0.003271
                    1  0.011454
                    2 -0.008182
                    0 -0.003271
                    1  0.011454
                    2 -0.008182

    Note: Everytime this code is run, the output will be different because
          "make_blobs" function generates random matrices.

    Parameters
    ----------
    copy : boolean (default = True)
        If True, then copies data then removes mean from data. False might
        cause data to be overwritten with its mean centered version.
    handle : cuml.Handle
        If it is None, a new one is created just for this class
    n_components : int (default = 1)
        The number of top K singular vectors / values you want.
        Must be <= number(columns).
    svd_solver : 'full'
        Only Full algorithm is supported.
    verbose : bool
        Whether to print debug spews
    whiten : boolean (default = False)
        If True, de-correlates the components. This is done by dividing them by
        the corresponding singular values then multiplying by sqrt(n_samples).
        Whitening allows each component to have unit variance and removes
        multi-collinearity. It might be beneficial for downstream
        tasks like LinearRegression where correlated features cause problems.


    Attributes
    ----------
    components_ : array
        The top K components (VT.T[:,:n_components]) in U, S, VT = svd(X)
    explained_variance_ : array
        How much each component explains the variance in the data given by S**2
    explained_variance_ratio_ : array
        How much in % the variance is explained given by S**2/sum(S**2)
    singular_values_ : array
        The top K singular values. Remember all singular values >= 0
    mean_ : array
        The column wise mean of X. Used to mean - center the data first.
    noise_variance_ : float
        From Bishop 1999's Textbook. Used in later tasks like calculating the
        estimated covariance of X.

    Notes
    ------
    PCA considers linear combinations of features, specifically those that
    maximise global variance structure. This means PCA is fantastic for global
    structure analyses, but weak for local relationships. Consider UMAP or
    T-SNE for a locally important embedding.

    **Applications of PCA**

        PCA is used extensively in practice for data visualization and data
        compression. It has been used to visualize extremely large word
        embeddings like Word2Vec and GloVe in 2 or 3 dimensions, large
        datasets of everyday objects and images, and used to distinguish
        between cancerous cells from healthy cells.


    For an additional example see `the PCA notebook
    <https://github.com/rapidsai/notebooks/blob/master/cuml/pca_demo.ipynb>`_.
    For additional docs, see `scikitlearn's PCA
    <http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_.
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
        try:
            from cuml.decomposition.pca_mg import PCAMG as cumlPCA
        except ImportError:
            raise Exception("cuML has not been built with multiGPU support "
                            "enabled. Build with the --multigpu flag to"
                            " enable multiGPU support.")

        handle = worker_state(sessionId)["handle"]
        return cumlPCA(handle=handle, **kwargs), dfs

    @staticmethod
    def _func_fit(f, M, N, partsToRanks, rank, transform):
        m, dfs = f
        return m.fit(dfs, M, N, partsToRanks, rank, transform)

    @staticmethod
    def _func_transform(f, df, M, N, partsToRanks, rank):
        m, dfs = f
        return m.transform(df, M, N, partsToRanks, rank)

    @staticmethod
    def _func_inverse_transform(f, df, M, N, partsToRanks, rank):        
        m, dfs = f
        return m.inverse_transform(df, M, N, partsToRanks, rank)

    @staticmethod
    def _func_get_first(f):
        return f[0]

    @staticmethod
    def _func_get_idx(f, idx):
        return f[idx]

    @staticmethod
    def _func_xform(model, df):       
        return model.transform(df)

    @staticmethod
    def _func_get_size(df):
        return df.shape[0]

    def fit(self, X, _transform=False):
        """
        Fit the model with X.

        Parameters
        ----------
        X : dask cuDF input

        """
        gpu_futures = self.client.sync(extract_ddf_partitions, X, agg=False)

        self.rnks = dict()
        rnk_counter = 0
        worker_to_parts = OrderedDict()
        for w, p in gpu_futures:
            if w not in worker_to_parts:
                worker_to_parts[w] = []
            if w not in self.rnks.keys():
                self.rnks[w] = rnk_counter
                rnk_counter = rnk_counter + 1
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
        self.pca_models = [(wf[0], self.client.submit(
            PCA._func_create_model,
            comms.sessionId,
            wf[1],
            **self.kwargs,
            workers=[wf[0]],
            key="%s-%s" % (key, idx)))
            for idx, wf in enumerate(worker_to_parts.items())]

        key = uuid1()
        pca_fit = dict([(worker_info[wf[0]]["r"], self.client.submit(
            PCA._func_fit,
            wf[1],
            M, N,
            partsToRanks,
            worker_info[wf[0]]["r"],
            _transform,
            key="%s-%s" % (key, idx),
            workers=[wf[0]]))
            for idx, wf in enumerate(self.pca_models)])

        wait(list(pca_fit.values()))
        raise_exception_from_futures(list(pca_fit.values()))

        comms.destroy()

        self.local_model = self.client.submit(PCA._func_get_first,
                                              self.pca_models[0][1]).result()

        self.components_ = self.local_model.components_
        self.explained_variance_ = self.local_model.explained_variance_
        self.explained_variance_ratio_ = self.local_model.explained_variance_ratio_
        self.singular_values_ = self.local_model.singular_values_
        self.noise_variance = self.local_model.noise_variance_

        out_futures = []
        if _transform:
            completed_part_map = {}
            for rank, size in partsToRanks:
                if rank not in completed_part_map:
                    completed_part_map[rank] = 0
           
                f = pca_fit[rank]
                out_futures.append(self.client.submit(
                    PCA._func_get_idx, f, completed_part_map[rank]))

                completed_part_map[rank] += 1

            return to_dask_cudf(out_futures)

    def _transform(self, X):
        gpu_futures = self.client.sync(extract_ddf_partitions, X, agg=False)

        worker_to_parts = OrderedDict()
        for w, p in gpu_futures:
            if w not in worker_to_parts:
                worker_to_parts[w] = []
            worker_to_parts[w].append(p)

        workers = list(map(lambda x: x[0], gpu_futures))

        key = uuid1()
        partsToRanks = [(self.rnks[wf[0]], self.client.submit(
            PCA._func_get_size,
            wf[1],
            workers=[wf[0]],
            key="%s-%s" % (key, idx)).result())
            for idx, wf in enumerate(gpu_futures)]

        N = X.shape[1]
        M = reduce(lambda a,b: a+b, map(lambda x: x[1], partsToRanks))

        key = uuid1()
        pca_transform = dict([(self.rnks[wf[0]], self.client.submit(
            PCA._func_transform,
            wf[1],
            worker_to_parts[wf[0]],
            M, N,
            partsToRanks,
            self.rnks[wf[0]],
            key="%s-%s" % (key, idx),
            workers=[wf[0]]))
            for idx, wf in enumerate(self.pca_models)])

        wait(list(pca_transform.values()))
        raise_exception_from_futures(list(pca_transform.values()))

        out_futures = []       
        completed_part_map = {}
        for rank, size in partsToRanks:
            if rank not in completed_part_map:
                completed_part_map[rank] = 0
           
            f = pca_transform[rank]
            out_futures.append(self.client.submit(
                PCA._func_get_idx, f, completed_part_map[rank]))

            completed_part_map[rank] += 1

        return to_dask_cudf(out_futures)

    def _inverse_transform(self, X):       
        gpu_futures = self.client.sync(extract_ddf_partitions, X, agg=False)

        worker_to_parts = OrderedDict()
        for w, p in gpu_futures:
            if w not in worker_to_parts:
                worker_to_parts[w] = []
            worker_to_parts[w].append(p)

        workers = list(map(lambda x: x[0], gpu_futures))

        key = uuid1()
        partsToRanks = [(self.rnks[wf[0]], self.client.submit(
            PCA._func_get_size,
            wf[1],
            workers=[wf[0]],
            key="%s-%s" % (key, idx)).result())
            for idx, wf in enumerate(gpu_futures)]

        N = X.shape[1]
        M = reduce(lambda a,b: a+b, map(lambda x: x[1], partsToRanks))

        key = uuid1()
        pca_inverse_transform = dict([(self.rnks[wf[0]], self.client.submit(
            PCA._func_inverse_transform,
            wf[1],
            worker_to_parts[wf[0]],
            M, N,
            partsToRanks,
            self.rnks[wf[0]],
            key="%s-%s" % (key, idx),
            workers=[wf[0]]))
            for idx, wf in enumerate(self.pca_models)])

        wait(list(pca_inverse_transform.values()))
        raise_exception_from_futures(list(pca_inverse_transform.values()))

        out_futures = []       
        completed_part_map = {}
        for rank, size in partsToRanks:
            if rank not in completed_part_map:
                completed_part_map[rank] = 0
           
            f = pca_inverse_transform[rank]
            out_futures.append(self.client.submit(
                PCA._func_get_idx, f, completed_part_map[rank]))

            completed_part_map[rank] += 1

        return to_dask_cudf(out_futures)

    def fit_transform(self, X):     
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : dask cuDF

        Returns
        -------
        X_new : dask cuDF
        """
        return self.fit(X, _transform=True)

    def transform(self, X):
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set.

        Parameters
        ----------
        X : dask cuDF

        Returns
        -------
        X_new : dask cuDF

        """
        return self._transform(X)

    def inverse_transform(self, X):
        """
        Transform data back to its original space.

        In other words, return an input X_original whose transform would be X.

        Parameters
        ----------
        X : dask cuDF

        Returns
        -------
        X_original : dask cuDF

        """
        return self._inverse_transform(X)

    def get_param_names(self):
        return list(self.kwargs.keys())

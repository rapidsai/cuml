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

from cuml.dask.common import raise_exception_from_futures
from cuml.dask.common.comms import worker_state, CommsContext

from cuml.dask.common.input_utils import to_output

from cuml.dask.common.utils import patch_cupy_sparse_serialization


from cuml.dask.common.input_utils import MGData


from dask.distributed import default_client
from dask.distributed import wait

from functools import reduce

from uuid import uuid1

from cuml.dask.common.base import DelayedTransformMixin
from cuml.dask.common.base import DelayedInverseTransformMixin


class PCA(DelayedTransformMixin, DelayedInverseTransformMixin):
    """
    PCA (Principal Component Analysis) is a fundamental dimensionality
    reduction technique used to combine features in X in linear combinations
    such that each new component captures the most information or variance of
    the data. N_components is usually small, say at 3, where it can be used for
    data visualization, data compression and exploratory analysis.

    cuML's multi-node multi-gpu (MNMG) PCA expects a dask cuDF input, and
    provides a "Full" algorithm. It uses a full eigendecomposition
    then selects the top K eigenvectors.

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
    handle : cuml.Handle
        If it is None, a new one is created just for this class
    n_components : int (default = 1)
        The number of top K singular vectors / values you want.
        Must be <= number(columns).
    svd_solver : 'full'
        Only Full algorithm is supported since it's significantly faster on GPU
        then the other solvers including randomized SVD.
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

        patch_cupy_sparse_serialization(self.client)

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
        print(str(f))
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

        X = self.client.persist(X)

        wait(X)

        data = MGData.single(data=X, client=self.client)
        self.datatype = data.datatype

        print(str(data.workers))

        comms = CommsContext(comms_p2p=False)
        comms.init(workers=data.workers)

        print("Initialized comms")

        data.calculate_parts_to_sizes(comms)

        print("Done parts to sizes")

        M = data.total_rows
        N = X.shape[1]


        key = uuid1()
        pca_models = [(wf[0], self.client.submit(
            PCA._func_create_model,
            comms.sessionId,
            wf[1],
            **self.kwargs,
            workers=[wf[0]],
            key="%s-%s" % (key, idx)))
            for idx, wf in enumerate(data.gpu_futures.items())]

        wait(pca_models)

        key = uuid1()
        pca_fit = dict([(data.worker_info[wf[0]]["r"], self.client.submit(
            PCA._func_fit,
            wf[1],
            M, N,
            data.parts_to_sizes[data.worker_info[wf[0]]["r"]],
            data.worker_info[wf[0]]["r"],
            _transform,
            key="%s-%s" % (key, idx),
            workers=[wf[0]]))
            for idx, wf in enumerate(pca_models)])

        wait(list(pca_fit.values()))
        raise_exception_from_futures(list(pca_fit.values()))

        comms.destroy()

        self.local_model = self.client.submit(PCA._func_get_first,
                                              pca_models[0][1]).result()

        self.components_ = self.local_model.components_
        self.explained_variance_ = self.local_model.explained_variance_
        self.explained_variance_ratio_ = \
            self.local_model.explained_variance_ratio_
        self.singular_values_ = self.local_model.singular_values_
        self.noise_variance = self.local_model.noise_variance_

        # TODO: Clean this up!
        out_futures = []
        if _transform:
            completed_part_map = {}
            for rank, size in data.parts_to_sizes:
                if rank not in completed_part_map:
                    completed_part_map[rank] = 0

                f = pca_fit[rank]
                out_futures.append(self.client.submit(
                    PCA._func_get_idx, f, completed_part_map[rank]))

                completed_part_map[rank] += 1

            return to_output(out_futures, self.datatype)

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

    def transform(self, X, delayed=True, parallel=5):
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
        return self._transform(X,
                               n_dims=2,
                               delayed=delayed,
                               parallel=parallel)

    def inverse_transform(self, X, delayed=True, parallel=5):
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
        return self._inverse_transform(X,
                                       n_dims=2,
                                       delayed=delayed,
                                       parallel=parallel)

    def get_param_names(self):
        return list(self.kwargs.keys())

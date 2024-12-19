# Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

from cuml.dask.decomposition.base import BaseDecomposition
from cuml.dask.decomposition.base import DecompositionSyncFitMixin

from cuml.dask.common.base import mnmg_import
from cuml.dask.common.base import DelayedTransformMixin
from cuml.dask.common.base import DelayedInverseTransformMixin


class PCA(
    BaseDecomposition,
    DelayedTransformMixin,
    DelayedInverseTransformMixin,
    DecompositionSyncFitMixin,
):
    """
    PCA (Principal Component Analysis) is a fundamental dimensionality
    reduction technique used to combine features in X in linear combinations
    such that each new component captures the most information or variance of
    the data. N_components is usually small, say at 3, where it can be used for
    data visualization, data compression and exploratory analysis.

    cuML's multi-node multi-gpu (MNMG) PCA expects a dask-cuDF object as input
    and provides 2 algorithms, Full and Jacobi. Full (default) uses a full
    eigendecomposition then selects the top K eigenvectors. The Jacobi
    algorithm can be much faster as it iteratively tries to correct the top K
    eigenvectors, but might be less accurate.

    Examples
    --------
    .. code-block:: python

        >>> from dask_cuda import LocalCUDACluster
        >>> from dask.distributed import Client, wait
        >>> import cupy as cp
        >>> from cuml.dask.decomposition import PCA
        >>> from cuml.dask.datasets import make_blobs

        >>> cluster = LocalCUDACluster(threads_per_worker=1)
        >>> client = Client(cluster)

        >>> nrows = 6
        >>> ncols = 3
        >>> n_parts = 2

        >>> X_cudf, _ = make_blobs(n_samples=nrows, n_features=ncols,
        ...                        centers=1, n_parts=n_parts,
        ...                        cluster_std=0.01, random_state=10,
        ...                        dtype=cp.float32)

        >>> blobs = X_cudf.compute()
        >>> print(blobs) # doctest: +SKIP
        [[8.688037  3.122401  1.2581943]
        [8.705028  3.1070278 1.2705998]
        [8.70239   3.1102846 1.2716919]
        [8.695665  3.1042147 1.2635932]
        [8.681095  3.0980906 1.2745825]
        [8.705454  3.100002  1.2657361]]

        >>> cumlModel = PCA(n_components = 1, whiten=False)
        >>> XT = cumlModel.fit_transform(X_cudf)
        >>> print(XT.compute()) # doctest: +SKIP
        [[-1.7516235e-02]
        [ 7.8094802e-03]
        [ 4.2757220e-03]
        [-6.7228684e-05]
        [-5.0618490e-03]
        [ 1.0557819e-02]]
        >>> client.close()
        >>> cluster.close()

    Parameters
    ----------
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    n_components : int (default = 1)
        The number of top K singular vectors / values you want.
        Must be <= number(columns).
    svd_solver : 'full', 'jacobi', 'auto'
        'full': Run exact full SVD and select the components by postprocessing
        'jacobi': Iteratively compute SVD of the covariance matrix
        'auto': For compatibility with Scikit-learn. Alias for 'jacobi'.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
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
    -----
    PCA considers linear combinations of features, specifically those that
    maximize global variance structure. This means PCA is fantastic for global
    structure analyses, but weak for local relationships. Consider UMAP or
    T-SNE for a locally important embedding.

    **Applications of PCA**

        PCA is used extensively in practice for data visualization and data
        compression. It has been used to visualize extremely large word
        embeddings like Word2Vec and GloVe in 2 or 3 dimensions, large
        datasets of everyday objects and images, and used to distinguish
        between cancerous cells from healthy cells.

    For additional docs, see `scikitlearn's PCA
    <http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_.
    """

    def __init__(self, *, client=None, verbose=False, **kwargs):

        super().__init__(
            model_func=PCA._create_pca,
            client=client,
            verbose=verbose,
            **kwargs,
        )

    def fit(self, X):
        """
        Fit the model with X.

        Parameters
        ----------
        X : dask cuDF input
        """

        self._fit(X)
        return self

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
        return self.fit(X).transform(X)

    def transform(self, X, delayed=True):
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
        return self._transform(X, n_dims=2, delayed=delayed)

    def inverse_transform(self, X, delayed=True):
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
        return self._inverse_transform(X, n_dims=2, delayed=delayed)

    def _get_param_names(self):
        return list(self.kwargs.keys())

    @staticmethod
    @mnmg_import
    def _create_pca(handle, datatype, **kwargs):
        from cuml.decomposition.pca_mg import PCAMG as cumlPCA

        return cumlPCA(handle=handle, output_type=datatype, **kwargs)

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


class TruncatedSVD(
    BaseDecomposition,
    DelayedTransformMixin,
    DelayedInverseTransformMixin,
    DecompositionSyncFitMixin,
):
    """
    Examples
    --------
    .. code-block:: python

        >>> from dask_cuda import LocalCUDACluster
        >>> from dask.distributed import Client, wait
        >>> import cupy as cp
        >>> from cuml.dask.decomposition import TruncatedSVD
        >>> from cuml.dask.datasets import make_blobs

        >>> cluster = LocalCUDACluster(threads_per_worker=1)
        >>> client = Client(cluster)

        >>> nrows = 6
        >>> ncols = 3
        >>> n_parts = 2

        >>> X_cudf, _ = make_blobs(n_samples=nrows, n_features=ncols,
        ...                        centers=1, n_parts=n_parts,
        ...                        cluster_std=1.8, random_state=10,
        ...                        dtype=cp.float32)
        >>> in_blobs = X_cudf.compute()
        >>> print(in_blobs) # doctest: +SKIP
        [[ 6.953966    6.2313757   0.84974563]
        [10.012338    3.4641726   3.0827546 ]
        [ 9.537406    4.0504313   3.2793145 ]
        [ 8.32713     2.957846    1.8215517 ]
        [ 5.7044296   1.855514    3.7996366 ]
        [10.089077    2.1995444   2.2072687 ]]
        >>> cumlModel = TruncatedSVD(n_components = 1)
        >>> XT = cumlModel.fit_transform(X_cudf)
        >>> result = XT.compute()
        >>> print(result) # doctest: +SKIP
        [[ 8.699628   0.         0.       ]
        [11.018815   0.         0.       ]
        [10.8554535  0.         0.       ]
        [ 9.000192   0.         0.       ]
        [ 6.7628784  0.         0.       ]
        [10.40526    0.         0.       ]]
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
    svd_solver : 'full', 'jacobi'
        Only Full algorithm is supported since it's significantly faster on GPU
        then the other solvers including randomized SVD.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

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

    """

    def __init__(self, *, client=None, **kwargs):
        """
        Constructor for distributed TruncatedSVD model
        """
        super().__init__(
            model_func=TruncatedSVD._create_tsvd, client=client, **kwargs
        )

    def fit(self, X, _transform=False):
        """
        Fit the model with X.

        Parameters
        ----------
        X : dask cuDF input

        """

        # `_transform=True` here as tSVD currently needs to
        # call `fit_transform` to be able to build
        # `explained_variance_`
        out = self._fit(X, _transform=True)
        if _transform:
            return out
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
        return self.fit(X, _transform=True)

    def transform(self, X, delayed=True):
        """
        Apply dimensionality reduction to `X`.

        `X` is projected on the first principal components previously extracted
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
    def _create_tsvd(handle, datatype, **kwargs):
        from cuml.decomposition.tsvd_mg import TSVDMG as cumlTSVD

        return cumlTSVD(handle=handle, output_type=datatype, **kwargs)

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

from cuml.dask.decomposition.base import BaseDecomposition
from cuml.dask.decomposition.base import DecompositionSyncFitMixin

from cuml.dask.common.base import mnmg_import
from cuml.dask.common.base import DelayedTransformMixin
from cuml.dask.common.base import DelayedInverseTransformMixin


class TruncatedSVD(BaseDecomposition,
                   DelayedTransformMixin,
                   DelayedInverseTransformMixin,
                   DecompositionSyncFitMixin):
    """
    Examples
    ---------

    .. code-block:: python

        from dask_cuda import LocalCUDACluster
        from dask.distributed import Client, wait
        import numpy as np
        from cuml.dask.decomposition import TruncatedSVD
        from cuml.dask.datasets import make_blobs

        cluster = LocalCUDACluster(threads_per_worker=1)
        client = Client(cluster)

        nrows = 6
        ncols = 3
        n_parts = 2

        X_cudf, _ = make_blobs(nrows, ncols, 1, n_parts,
                        cluster_std=1.8, verbose=False,
                        random_state=10, dtype=np.float32)

        wait(X_cudf)

        print("Input Matrix")
        print(X_cudf.compute())

        cumlModel = TruncatedSVD(n_components = 1)
        XT = cumlModel.fit_transform(X_cudf)

        print("Transformed Input Matrix")
        print(XT.compute())

    Output:

    .. code-block:: python

          Input Matrix:
                              0         1          2
                    0 -8.519647 -8.519222  -8.865648
                    1 -6.107700 -8.350124 -10.351215
                    2 -8.026635 -9.442240  -7.561770
                    0 -8.519647 -8.519222  -8.865648
                    1 -6.107700 -8.350124 -10.351215
                    2 -8.026635 -9.442240  -7.561770

          Transformed Input Matrix:
                               0
                    0  14.928891
                    1  14.487295
                    2  14.431235
                    0  14.928891
                    1  14.487295
                    2  14.431235
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

    def __init__(self, client=None, **kwargs):
        """
        Constructor for distributed TruncatedSVD model
        """
        super(TruncatedSVD, self).__init__(TruncatedSVD._create_tsvd,
                                           client,
                                           **kwargs)

    def fit(self, X, _transform=False):
        """
        Fit the model with X.

        Parameters
        ----------
        X : dask cuDF input

        """
        return self._fit(X, _transform)

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
                               delayed=delayed)

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
        return self._inverse_transform(X,
                                       n_dims=2,
                                       delayed=delayed)

    def get_param_names(self):
        return list(self.kwargs.keys())

    @staticmethod
    @mnmg_import
    def _create_tsvd(handle, datatype, **kwargs):
        from cuml.decomposition.tsvd_mg import TSVDMG as cumlTSVD
        return cumlTSVD(handle=handle, output_type=datatype, **kwargs)

# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

from cuml.dask.common.base import BaseEstimator, DelayedTransformMixin
from cuml.dask.common.input_utils import DistributedDataHandler


class UMAP(BaseEstimator, DelayedTransformMixin):
    """
    Uniform Manifold Approximation and Projection

    Finds a low dimensional embedding of the data that approximates
    an underlying manifold.

    Adapted from https://github.com/lmcinnes/umap/blob/master/umap/umap_.py

    Examples
    --------
    .. code-block:: python

        >>> from dask_cuda import LocalCUDACluster
        >>> from dask.distributed import Client
        >>> import dask.array as da
        >>> from cuml.datasets import make_blobs
        >>> from cuml.manifold import UMAP
        >>> from cuml.dask.manifold import UMAP as MNMG_UMAP
        >>> import numpy as np

        >>> cluster = LocalCUDACluster(threads_per_worker=1)
        >>> client = Client(cluster)

        >>> X, y = make_blobs(1000, 10, centers=42, cluster_std=0.1,
        ...                   dtype=np.float32, random_state=10)

        >>> local_model = UMAP(random_state=10, verbose=0)

        >>> selection = np.random.RandomState(10).choice(1000, 100)
        >>> X_train = X[selection]
        >>> y_train = y[selection]
        >>> local_model.fit(X_train, y=y_train)
        UMAP()

        >>> distributed_model = MNMG_UMAP(model=local_model)
        >>> distributed_X = da.from_array(X, chunks=(500, -1))
        >>> embedding = distributed_model.transform(distributed_X)
        >>> result = embedding.compute()
        >>> print(result) # doctest: +SKIP
        [[  4.1684933    4.1890593 ]
        [  5.0110254   -5.2143383 ]
        [  1.7776365  -17.665699  ]
        ...
        [ -6.6378727   -0.15353012]
        [ -3.1891193   -0.83906937]
        [ -0.5042019    2.1454725 ]]
        >>> client.close()
        >>> cluster.close()

    Notes
    -----
    This module is heavily based on Leland McInnes' reference UMAP package
    [1]_.

    However, there are a number of differences and features that are
    not yet implemented in `cuml.umap`:

    * Using a non-Euclidean distance metric (support for a fixed set
      of non-Euclidean metrics is planned for an upcoming release).
    * Using a pre-computed pairwise distance matrix (under consideration
      for future releases)
    * Manual initialization of initial embedding positions

    In addition to these missing features, you should expect to see
    the final embeddings differing between `cuml.umap` and the reference
    UMAP.

    **Known issue:** If a UMAP model has not yet been fit, it cannot be pickled

    References
    ----------
    .. [1] `Leland McInnes, John Healy, James Melville
       UMAP: Uniform Manifold Approximation and Projection for Dimension
       Reduction. <https://arxiv.org/abs/1802.03426>`_

    """

    def __init__(self, *, model, client=None, **kwargs):
        super().__init__(client=client, **kwargs)

        self._set_internal_model(model)

    def transform(self, X, convert_dtype=True):
        """
        Transform X into the existing embedded space and return that
        transformed output.

        Please refer to the reference UMAP implementation for information
        on the differences between fit_transform() and running fit()
        transform().

        Specifically, the transform() function is stochastic:
        https://github.com/lmcinnes/umap/issues/158

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            New data to be transformed.
            Acceptable formats: dask cuDF, dask CuPy/NumPy/Numba Array

        Returns
        -------
        X_new : array, shape (n_samples, n_components)
            Embedding of the new data in low-dimensional space.
        """
        data = DistributedDataHandler.create(data=X, client=self.client)
        self.datatype = data.datatype
        return self._transform(X, convert_dtype=convert_dtype)

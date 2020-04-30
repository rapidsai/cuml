# Copyright (c) 2020, NVIDIA CORPORATION.
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


class UMAP(BaseEstimator,
           DelayedTransformMixin):
    def __init__(self, model, client=None, **kwargs):
        """Uniform Manifold Approximation and Projection
        Finds a low dimensional embedding of the data that approximates
        an underlying manifold.

        Adapted from https://github.com/lmcinnes/umap/blob/master/umap/umap_.py

        Examples
        ----------

        .. code-block:: python

            from dask_cuda import LocalCUDACluster
            from dask.distributed import Client
            from cuml.dask.datasets import make_blobs
            from cuml.manifold import UMAP
            from cuml.dask.manifold import UMAP as MNMG_UMAP
            import numpy as np

            cluster = LocalCUDACluster(threads_per_worker=1)
            client = Client(cluster)

            X, y = make_blobs(1000, 10,
                            centers=42,
                            cluster_std=0.1,
                            dtype=np.float32,
                            n_parts=2,
                            output='array')

            local_model = UMAP()

            selection = np.random.choice(1000, 100)
            X_train = X[selection].compute()
            y_train = y[selection].compute()

            local_model.fit(X_train, y=y_train)

            distributed_model = MNMG_UMAP(local_model)
            embedding = distributed_model.transform(X)

        Note: Everytime this code is run, the output will be different because
            "make_blobs" function generates random matrices.

        Notes
        -----
        This module is heavily based on Leland McInnes' reference UMAP package.
        However, there are a number of differences and features that are
        not yet implemented in cuml.umap:
        * Using a non-Euclidean distance metric (support for a fixed set
            of non-Euclidean metrics is planned for an upcoming release).
        * Using a pre-computed pairwise distance matrix (under consideration
            for future releases)
        * Manual initialization of initial embedding positions

        In addition to these missing features, you should expect to see
        the final embeddings differing between cuml.umap and the reference
        UMAP. In particular, the reference UMAP uses an approximate kNN
        algorithm for large data sizes while cuml.umap always uses exact
        kNN.

        Known issue: If a UMAP model has not yet been fit, it cannot be pickled
        However, after fitting, a UMAP mode.

        References
        ----------
        * Leland McInnes, John Healy, James Melville
        UMAP: Uniform Manifold Approximation and Projection for Dimension
        Reduction
        https://arxiv.org/abs/1802.03426

        """
        super(UMAP, self).__init__(client, **kwargs)
        self.local_model = model

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
        return self._transform(X,
                               convert_dtype=convert_dtype)

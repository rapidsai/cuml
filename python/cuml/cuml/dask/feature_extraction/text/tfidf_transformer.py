#
# Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

import dask
from toolz import first
import dask.array

from cuml.common import with_cupy_rmm
from cuml.dask.common.base import BaseEstimator
from cuml.dask.common.base import DelayedTransformMixin
from cuml.dask.common.utils import wait_and_raise_from_futures
from cuml.dask.common.func import reduce
from cuml.dask.common.input_utils import DistributedDataHandler

from cuml.feature_extraction.text import TfidfTransformer as s_TfidfTransformer


class TfidfTransformer(BaseEstimator, DelayedTransformMixin):

    """
    Distributed TF-IDF  transformer

    Examples
    --------
    .. code-block:: python

        >>> import cupy as cp
        >>> from sklearn.datasets import fetch_20newsgroups
        >>> from sklearn.feature_extraction.text import CountVectorizer
        >>> from dask_cuda import LocalCUDACluster
        >>> from dask.distributed import Client
        >>> from cuml.dask.common import to_sparse_dask_array
        >>> from cuml.dask.naive_bayes import MultinomialNB
        >>> import dask
        >>> from cuml.dask.feature_extraction.text import TfidfTransformer

        >>> # Create a local CUDA cluster
        >>> cluster = LocalCUDACluster()
        >>> client = Client(cluster)

        >>> # Load corpus
        >>> twenty_train = fetch_20newsgroups(subset='train',
        ...                         shuffle=True, random_state=42)
        >>> cv = CountVectorizer()
        >>> xformed = cv.fit_transform(twenty_train.data).astype(cp.float32)
        >>> X = to_sparse_dask_array(xformed, client)

        >>> y = dask.array.from_array(twenty_train.target, asarray=False,
        ...                     fancy=False).astype(cp.int32)

        >>> multi_gpu_transformer = TfidfTransformer()
        >>> X_transformed = multi_gpu_transformer.fit_transform(X)
        >>> X_transformed.compute_chunk_sizes()
        dask.array<...>

        >>> model = MultinomialNB()
        >>> model.fit(X_transformed, y)
        <cuml.dask.naive_bayes.naive_bayes.MultinomialNB object at 0x...>
        >>> result = model.score(X_transformed, y)
        >>> print(result) # doctest: +SKIP
        array(0.93264981)
        >>> client.close()
        >>> cluster.close()

    """

    def __init__(self, *, client=None, verbose=False, **kwargs):

        """
        Create new  distributed TF-IDF transformer instance

        Parameters
        -----------

        client : dask.distributed.Client optional Dask client to use
        """
        super().__init__(client=client, verbose=verbose, **kwargs)

        self.datatype = "cupy"

        # Make any potential model args available and catch any potential
        # ValueErrors before distributed training begins.
        self._set_internal_model(s_TfidfTransformer(**kwargs))

    @staticmethod
    @with_cupy_rmm
    def _set_doc_stats(X, kwargs):
        model = s_TfidfTransformer(**kwargs)
        # Below is only required if we have to set stats
        if model.use_idf:
            model._set_doc_stats(X)

        return model

    @staticmethod
    def _merge_stats_to_model(models):
        modela = first(models)
        if modela.use_idf:
            for model in models[1:]:
                modela.__n_samples += model.__n_samples
                modela.__df += model.__df
        return modela

    @staticmethod
    def _set_idf_diag(model):
        model._set_idf_diag()
        return model

    @with_cupy_rmm
    def fit(self, X, y=None):

        """
        Fit distributed TFIDF Transformer

        Parameters
        ----------

        X : dask.Array with blocks containing dense or sparse cupy arrays

        Returns
        -------

        cuml.dask.feature_extraction.text.TfidfTransformer instance
        """
        # Only Dask.Array supported for now
        if not isinstance(X, dask.array.core.Array):
            raise ValueError("Only dask.Array is supported for X")

        if len(X.chunks[1]) != 1:
            raise ValueError(
                "X must be chunked by row only. "
                "Multi-dimensional chunking is not supported"
            )

        # We don't' do anything if we don't need idf
        if not self.internal_model.use_idf:
            return self

        futures = DistributedDataHandler.create(X, self.client)

        models = [
            self.client.submit(
                self._set_doc_stats, part, self.kwargs, pure=False
            )
            for w, part in futures.gpu_futures
        ]

        models = reduce(models, self._merge_stats_to_model, client=self.client)

        wait_and_raise_from_futures([models])

        models = self.client.submit(self._set_idf_diag, models, pure=False)

        wait_and_raise_from_futures([models])

        self._set_internal_model(models)

        return self

    @staticmethod
    def _get_part(parts, idx):
        return parts[idx]

    @staticmethod
    def _get_size(arrs):
        return arrs.shape[0]

    def fit_transform(self, X, y=None):
        """
        Fit distributed TFIDFTransformer and then transform
        the given set of data samples.

        Parameters
        ----------

        X : dask.Array with blocks containing dense or sparse cupy arrays

        Returns
        -------

        dask.Array with blocks containing transformed sparse cupy arrays

        """
        return self.fit(X).transform(X)

    def transform(self, X, y=None):
        """
        Use distributed TFIDFTransformer to transform the
        given set of data samples.

        Parameters
        ----------

        X : dask.Array with blocks containing dense or sparse cupy arrays

        Returns
        -------

        dask.Array with blocks containing transformed sparse cupy arrays

        """
        if not isinstance(X, dask.array.core.Array):
            raise ValueError("Only dask.Array is supported for X")

        return self._transform(
            X, n_dims=2, delayed=True, output_collection_type="cupy"
        )

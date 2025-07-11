#
# Copyright (c) 2025, NVIDIA CORPORATION.
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


import cupy as cp
import numpy as np

from cython.operator cimport dereference as deref
from libc.stdint cimport uint64_t, uintptr_t

from pylibraft.common.handle import Handle

from libcpp cimport bool
from pylibraft.common.cpp.mdspan cimport (
    col_major,
    device_matrix_view,
    make_device_matrix_view,
    row_major,
)
from pylibraft.common.handle cimport device_resources

import cuml
from cuml.common import input_to_cuml_array
from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from cuml.internals.mixins import CMajorInputTagMixin, SparseInputTagMixin
from cuml.internals.utils import check_random_seed


cdef extern from "cuml/manifold/spectral_embedding.hpp" namespace "ML::SpectralEmbedding":

    cdef cppclass params:
        int n_components
        int n_neighbors
        bool norm_laplacian
        bool drop_first
        uint64_t seed

    cdef int spectral_embedding_cuvs(
        const device_resources &handle,
        params config,
        device_matrix_view[float, int, row_major] dataset,
        device_matrix_view[float, int, col_major] embedding) except +

cdef params config


@cuml.internals.api_return_array(get_output_type=True)
def spectral_embedding(A,
                       *,
                       n_components=8,
                       random_state=None,
                       n_neighbors=None,
                       norm_laplacian=True,
                       drop_first=True,
                       handle=None):
    """Project the sample on the first eigenvectors of the graph Laplacian.

    The adjacency matrix is used to compute a normalized graph Laplacian
    whose spectrum (especially the eigenvectors associated to the
    smallest eigenvalues) has an interpretation in terms of minimal
    number of cuts necessary to split the graph into comparably sized
    components.

    Note : Laplacian Eigenmaps is the actual algorithm implemented here.

    Parameters
    ----------
    A : array-like of shape (n_samples, n_features)
        The input data. A k-NN graph will be constructed.

    n_components : int, default=8
        The dimension of the projection subspace.

    random_state : int, RandomState instance or None, default=None
        A pseudo random number generator used for the initialization.
        Use an int to make the results deterministic across calls.

    n_neighbors : int or None, default=None
        Number of nearest neighbors for nearest_neighbors graph building.
        If None, n_neighbors will be set to max(n_samples/10, 1).
        Only used when A has shape (n_samples, n_features).

    norm_laplacian : bool, default=True
        If True, then compute symmetric normalized Laplacian.

    drop_first : bool, default=True
        Whether to drop the first eigenvector. For spectral embedding, this
        should be True as the first eigenvector should be constant vector for
        connected graph, but for spectral clustering, this should be kept as
        False to retain the first eigenvector.

    Returns
    -------
    embedding : cupy.ndarray of shape (n_samples, n_components)
        The reduced samples.

    Notes
    -----
    Spectral Embedding (Laplacian Eigenmaps) is most useful when the graph
    has one connected component. If there graph has many components, the first
    few eigenvectors will simply uncover the connected components of the graph.

    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.manifold import spectral_embedding
    >>> X = cp.random.rand(100, 20, dtype=cp.float32)
    >>> embedding = spectral_embedding(X, n_components=2, random_state=42)
    >>> embedding.shape
    (100, 2)
    """

    if handle is None:
        handle = Handle()
    cdef device_resources *h = <device_resources*><size_t>handle.getHandle()

    A, _n_rows, _n_cols, _ = \
        input_to_cuml_array(A, order="C", check_dtype=np.float32,
                            convert_to_dtype=cp.float32)
    A_ptr = <uintptr_t>A.ptr

    config.n_components = n_components
    config.seed = check_random_seed(random_state)

    config.n_neighbors = (
        n_neighbors
        if n_neighbors is not None
        else max(int(A.shape[0] / 10), 1)
    )

    config.norm_laplacian = norm_laplacian
    config.drop_first = drop_first

    if config.drop_first:
        config.n_components += 1

    eigenvectors = CumlArray.empty((A.shape[0], n_components), dtype=A.dtype, order='F')

    eigenvectors_ptr = <uintptr_t>eigenvectors.ptr

    cdef int _result = spectral_embedding_cuvs(
        deref(h), config,
        make_device_matrix_view[float, int, row_major](
            <float *>A_ptr, <int> A.shape[0], <int> A.shape[1]),
        make_device_matrix_view[float, int, col_major](
            <float *>eigenvectors_ptr, <int> A.shape[0], <int> n_components))

    return eigenvectors


class SpectralEmbedding(Base,
                        CMajorInputTagMixin,
                        SparseInputTagMixin):
    """Spectral embedding for non-linear dimensionality reduction.

    Forms an affinity matrix given by the specified function and
    applies spectral decomposition to the corresponding graph laplacian.
    The resulting transformation is given by the value of the
    eigenvectors for each data point.

    Note : Laplacian Eigenmaps is the actual algorithm implemented here.

    Parameters
    ----------
    n_components : int, default=2
        The dimension of the projected subspace.

    random_state : int, RandomState instance or None, default=None
        A pseudo random number generator used for the initialization.
        Use an int to make the results deterministic across calls.

    n_neighbors : int or None, default=None
        Number of nearest neighbors for nearest_neighbors graph building.
        If None, n_neighbors will be set to max(n_samples/10, 1).

    Attributes
    ----------
    embedding_ : cupy.ndarray of shape (n_samples, n_components)
        Spectral embedding of the training matrix.

    Notes
    -----
    Currently, cuML's SpectralEmbedding only supports the 'nearest_neighbors'
    affinity mode, where a k-NN graph is constructed from the input data.

    Spectral Embedding (Laplacian Eigenmaps) is most useful when the graph
    has one connected component. If there graph has many components, the first
    few eigenvectors will simply uncover the connected components of the graph.

    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.manifold import SpectralEmbedding
    >>> X = cp.random.rand(100, 20, dtype=cp.float32)
    >>> embedding = SpectralEmbedding(n_components=2, random_state=42)
    >>> X_transformed = embedding.fit_transform(X)
    >>> X_transformed.shape
    (100, 2)
    """

    def __init__(self, n_components=2, random_state=None, n_neighbors=None,
                 handle=None):
        super().__init__(handle=handle)
        self.n_components = n_components
        self.random_state = random_state
        self.n_neighbors = n_neighbors

    def fit_transform(self, X, y=None):
        """Fit the model from data in X and transform X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        X_new : cupy.ndarray of shape (n_samples, n_components)
            Spectral embedding of the training matrix.
        """
        self.fit(X, y)
        return self.embedding_

    def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self.embedding_ = self._fit(X, self.n_components,
                                    random_state=self.random_state,
                                    n_neighbors=self.n_neighbors)
        return self

    def _fit(self, A, n_components, random_state=None, n_neighbors=None):
        return spectral_embedding(A,
                                  n_components=n_components,
                                  random_state=random_state,
                                  n_neighbors=n_neighbors)

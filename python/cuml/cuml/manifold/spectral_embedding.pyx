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
import cupyx.scipy.sparse as cp_sp
import numpy as np
import scipy.sparse as sp
from pylibraft.common.handle import Handle

import cuml
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from cuml.internals.input_utils import input_to_cupy_array
from cuml.internals.interop import (
    InteropMixin,
    UnsupportedOnGPU,
    to_cpu,
    to_gpu,
)
from cuml.internals.mixins import CMajorInputTagMixin
from cuml.internals.utils import check_random_seed

from libc.stdint cimport uint64_t, uintptr_t
from libcpp cimport bool
from pylibraft.common.cpp.mdspan cimport (
    col_major,
    device_matrix_view,
    device_vector_view,
    make_device_matrix_view,
    make_device_vector_view,
    row_major,
)
from pylibraft.common.handle cimport device_resources


cdef extern from "cuml/manifold/spectral_embedding.hpp" \
        namespace "ML::SpectralEmbedding" nogil:

    cdef cppclass params:
        int n_components
        int n_neighbors
        bool norm_laplacian
        bool drop_first
        uint64_t seed

    cdef void transform(
        const device_resources &handle,
        params config,
        device_matrix_view[float, int, row_major] dataset,
        device_matrix_view[float, int, col_major] embedding) except +

    cdef void transform(
        const device_resources &handle,
        params config,
        device_vector_view[int, int] rows,
        device_vector_view[int, int] cols,
        device_vector_view[float, int] vals,
        device_matrix_view[float, int, col_major] embedding) except +


@cuml.internals.api_return_array(get_output_type=True)
def spectral_embedding(A,
                       *,
                       int n_components=8,
                       affinity="nearest_neighbors",
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
    A : array-like or sparse matrix of shape (n_samples, n_features) or \
        (n_samples, n_samples)
        If affinity is 'nearest_neighbors', this is the input data and a k-NN
        graph will be constructed. If affinity is 'precomputed', this is the
        affinity matrix. Supported formats for precomputed affinity: scipy
        sparse (CSR, CSC, COO), cupy sparse (CSR, CSC, COO), dense numpy
        arrays, or dense cupy arrays.
    n_components : int, default=8
        The dimension of the projection subspace.
    affinity : {'nearest_neighbors', 'precomputed'}, default='nearest_neighbors'
        How to construct the affinity matrix.
         - 'nearest_neighbors' : construct the affinity matrix by computing a
           graph of nearest neighbors.
         - 'precomputed' : interpret ``A`` as a precomputed affinity matrix.
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
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.

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

    cdef float* affinity_data_ptr = NULL
    cdef int* affinity_rows_ptr = NULL
    cdef int* affinity_cols_ptr = NULL
    cdef int affinity_nnz = 0

    if affinity == "nearest_neighbors":
        A = input_to_cupy_array(
            A, order="C", check_dtype=np.float32, convert_to_dtype=cp.float32
        ).array

        affinity_data_ptr = <float*><uintptr_t>A.data.ptr

        isfinite = cp.isfinite(A).all()
    elif affinity == "precomputed":
        # Coerce `A` to a canonical float32 COO sparse matrix
        if cp_sp.issparse(A):
            A = A.tocoo()
            if A.dtype != np.float32:
                A = A.astype("float32")
        elif sp.issparse(A):
            A = cp_sp.coo_matrix(A, dtype="float32")
        else:
            A = cp_sp.coo_matrix(cp.asarray(A, dtype="float32"))
        A.sum_duplicates()

        affinity_data = A.data
        affinity_rows = A.row
        affinity_cols = A.col
        affinity_nnz = A.nnz

        # laplacian kernel expects diagonal to be zero
        # remove diagonal elements since they are ignored in laplacian calculation anyway
        valid = affinity_rows != affinity_cols
        if not valid.all():
            affinity_data = affinity_data[valid]
            affinity_rows = affinity_rows[valid]
            affinity_cols = affinity_cols[valid]
            affinity_nnz = len(affinity_data)

        affinity_data_ptr = <float*><uintptr_t>affinity_data.data.ptr
        affinity_rows_ptr = <int*><uintptr_t>affinity_rows.data.ptr
        affinity_cols_ptr = <int*><uintptr_t>affinity_cols.data.ptr

        isfinite = cp.isfinite(affinity_data).all()
    else:
        raise ValueError(
            f"`affinity={affinity!r}` is not supported, expected one of "
            "['nearest_neighbors', 'precomputed']"
        )

    cdef int n_samples, n_features
    n_samples, n_features = A.shape

    if not isfinite:
        raise ValueError(
            "Input contains NaN or inf; nonfinite values are not supported"
        )

    if n_samples < 2:
        raise ValueError(
            f"Found array with {n_samples} sample(s) (shape={A.shape}) while a "
            f"minimum of 2 is required."
        )
    if n_features < 2:
        raise ValueError(
            f"Found array with {n_features} feature(s) (shape={A.shape}) while "
            f"a minimum of 2 is required."
        )

    # Allocate output array
    eigenvectors = CumlArray.empty(
        (A.shape[0], n_components), dtype=np.float32, order='F'
    )

    cdef params config
    config.seed = check_random_seed(random_state)
    config.norm_laplacian = norm_laplacian
    config.drop_first = drop_first
    config.n_components = n_components + 1 if drop_first else n_components
    config.n_neighbors = (
        n_neighbors
        if n_neighbors is not None
        else max(int(A.shape[0] / 10), 1)
    )
    cdef float* eigenvectors_ptr = <float *><uintptr_t>eigenvectors.ptr
    cdef bool precomputed = affinity == "precomputed"
    cdef device_resources *handle_ = <device_resources*><size_t>handle.getHandle()

    with nogil:
        if precomputed:
            transform(
                handle_[0],
                config,
                make_device_vector_view(affinity_rows_ptr, affinity_nnz),
                make_device_vector_view(affinity_cols_ptr, affinity_nnz),
                make_device_vector_view(affinity_data_ptr, affinity_nnz),
                make_device_matrix_view[float, int, col_major](
                    eigenvectors_ptr, n_samples, n_components,
                )
            )
        else:
            transform(
                handle_[0],
                config,
                make_device_matrix_view[float, int, row_major](
                    affinity_data_ptr, n_samples, n_features,
                ),
                make_device_matrix_view[float, int, col_major](
                    eigenvectors_ptr, n_samples, n_components,
                )
            )

    return eigenvectors


class SpectralEmbedding(Base,
                        InteropMixin,
                        CMajorInputTagMixin):
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
    affinity : {'nearest_neighbors', 'precomputed'}, default='nearest_neighbors'
        How to construct the affinity matrix.
         - 'nearest_neighbors' : construct the affinity matrix by computing a
           graph of nearest neighbors.
         - 'precomputed' : interpret ``X`` as a precomputed affinity matrix.
    random_state : int, RandomState instance or None, default=None
        A pseudo random number generator used for the initialization.
        Use an int to make the results deterministic across calls.
    n_neighbors : int or None, default=2
        Number of nearest neighbors for nearest_neighbors graph building.
        If None, n_neighbors will be set to max(n_samples/10, 1).
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

    Attributes
    ----------
    embedding_ : cupy.ndarray of shape (n_samples, n_components)
        Spectral embedding of the training matrix.
    n_neighbors_ : int
        Number of nearest neighbors effectively used.

    Notes
    -----
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
    _cpu_class_path = "sklearn.manifold.SpectralEmbedding"
    embedding_ = CumlArrayDescriptor(order="F")

    def __init__(self, n_components=2, affinity="nearest_neighbors",
                 random_state=None, n_neighbors=None,
                 handle=None, verbose=False, output_type=None):
        super().__init__(handle=handle, verbose=verbose, output_type=output_type)
        self.n_components = n_components
        self.affinity = affinity
        self.random_state = random_state
        self.n_neighbors = n_neighbors

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + [
            "n_components",
            "affinity",
            "random_state",
            "n_neighbors"
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        if model.affinity not in ("nearest_neighbors", "precomputed"):
            raise UnsupportedOnGPU(f"affinity={model.affinity!r} is not supported on GPU")
        params = {
            "n_components": model.n_components,
            "affinity": model.affinity,
            "random_state": model.random_state,
            "n_neighbors": model.n_neighbors
        }
        return params

    def _params_to_cpu(self):
        params = {
            "n_components": self.n_components,
            "affinity": self.affinity,
            "random_state": self.random_state,
            "n_neighbors": self.n_neighbors
        }
        return params

    def _attrs_from_cpu(self, model):
        return {
            "n_neighbors_": model.n_neighbors_,
            "embedding_": to_gpu(model.embedding_, order="F"),
            **super()._attrs_from_cpu(model)
        }

    def _attrs_to_cpu(self, model):
        return {
            "n_neighbors_": self.n_neighbors_,
            "embedding_": to_cpu(self.embedding_, order="F"),
            **super()._attrs_to_cpu(model),
        }

    def fit_transform(self, X, y=None) -> CumlArray:
        """Fit the model from data in X and transform X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features) or \
            (n_samples, n_samples)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features. If affinity is
            'precomputed', X is the affinity matrix. Supported formats for
            precomputed affinity: scipy sparse (CSR, CSC, COO), cupy sparse
            (CSR, CSC, COO), dense numpy arrays, or dense cupy arrays.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        X_new : cupy.ndarray of shape (n_samples, n_components)
            Spectral embedding of the training matrix.
        """
        self.fit(X, y)
        return self.embedding_

    def fit(self, X, y=None) -> "SpectralEmbedding":
        """Fit the model from data in X.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features) or \
            (n_samples, n_samples)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features. If affinity is
            'precomputed', X is the affinity matrix. Supported formats for
            precomputed affinity: scipy sparse (CSR, CSC, COO), cupy sparse
            (CSR, CSC, COO), dense numpy arrays, or dense cupy arrays.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        # Store n_neighbors_ for sklearn compatibility
        self.n_neighbors_ = (
            self.n_neighbors
            if self.n_neighbors is not None
            else max(int(X.shape[0] / 10), 1)
        )

        self.embedding_ = spectral_embedding(
            X,
            n_components=self.n_components,
            affinity=self.affinity,
            random_state=self.random_state,
            n_neighbors=self.n_neighbors_,
            handle=self.handle
        )

        return self

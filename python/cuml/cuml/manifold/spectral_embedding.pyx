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
from libcpp cimport bool

from pylibraft.common.handle import Handle

from pylibraft.common.cpp.mdspan cimport (
    col_major,
    device_matrix_view,
    make_device_matrix_view,
    row_major,
)
from pylibraft.common.handle cimport device_resources

import cuml
from cuml.common import input_to_cuml_array
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from cuml.internals.interop import InteropMixin, UnsupportedOnGPU
from cuml.internals.mixins import CMajorInputTagMixin, SparseInputTagMixin
from cuml.internals.utils import check_random_seed


cdef extern from "cuml/manifold/spectral_embedding.hpp" namespace "ML::SpectralEmbedding":

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
    if n_components <= 0:
        raise ValueError(f"n_components must be > 0. Got {n_components}")

    # Convert sparse input to dense (cuML doesn't support sparse directly)
    try:
        import scipy.sparse as sp
        if sp.issparse(A):
            # Convert sparse to dense automatically
            A = A.toarray()
    except ImportError:
        pass

    # Check for 1D input
    if hasattr(A, 'ndim') and A.ndim == 1:
        raise ValueError(
            "Expected 2D array, got 1D array instead:\n"
            f"array={A}.\n"
            "Reshape your data either using array.reshape(-1, 1) if "
            "your data has a single feature or array.reshape(1, -1) "
            "if it contains a single sample."
        )
    elif hasattr(A, 'shape') and len(A.shape) == 1:
        raise ValueError(
            "Expected 2D array, got 1D array instead:\n"
            f"array shape={A.shape}.\n"
            "Reshape your data either using array.reshape(-1, 1) if "
            "your data has a single feature or array.reshape(1, -1) "
            "if it contains a single sample."
        )

    # Check for complex data
    if hasattr(A, 'dtype'):
        if np.iscomplexobj(A):
            raise ValueError("Complex data not supported")

    # Check for empty data
    if hasattr(A, 'shape'):
        if len(A.shape) >= 2:
            if A.shape[0] == 0:
                raise ValueError(
                    f"Found array with 0 sample(s) (shape={A.shape}) while a "
                    f"minimum of 1 is required."
                )
            if A.shape[1] == 0:
                raise ValueError(
                    f"Found array with 0 feature(s) (shape={A.shape}) while a "
                    f"minimum of 1 is required."
                )
            # SpectralEmbedding requires at least 2 samples
            if A.shape[0] < 2:
                raise ValueError(
                    f"SpectralEmbedding requires at least 2 samples, but got "
                    f"only {A.shape[0]} sample(s)."
                )
            # SpectralEmbedding requires at least 2 features for meaningful embedding
            if A.shape[1] < 2:
                raise ValueError(
                    f"SpectralEmbedding requires at least 2 features for meaningful "
                    f"results, but got only {A.shape[1]} feature(s)."
                )

    # Check for NaN and Inf values
    if hasattr(A, 'dtype'):
        # Check if it's a numeric dtype (not object)
        if A.dtype != object and not (hasattr(A.dtype, 'name') and A.dtype.name == 'object'):
            # Convert to numpy array if needed for checking
            A_check = A
            if hasattr(A, 'to_numpy'):  # cudf DataFrame
                A_check = A.to_numpy()
            elif hasattr(A, 'values'):  # pandas DataFrame
                A_check = A.values
            elif hasattr(A, '__cuda_array_interface__'):  # cupy array
                A_check = cp.asnumpy(A)

            # Check for NaN and Inf
            A_np = np.asarray(A_check)
            if np.any(np.isnan(A_np)):
                raise ValueError(
                    "Input contains NaN"
                )
            if np.any(np.isinf(A_np)):
                raise ValueError(
                    "Input contains infinity or a value too large for dtype('float64')."
                )

    # Check for object dtype and validate contents
    if hasattr(A, 'dtype'):
        if A.dtype == object or (hasattr(A.dtype, 'name') and A.dtype.name == 'object'):
            # Try to convert to numeric array
            try:
                # First check if any element is not numeric
                flat_A = np.asarray(A).ravel()
                for elem in flat_A:
                    # Check if element is a dict, list, or other non-numeric type
                    if isinstance(elem, (dict, list, tuple, str)) or not np.isscalar(elem):
                        raise TypeError(
                            "argument must be a string or a real number, not "
                            f"'{type(elem).__name__}'"
                        )
                # If all elements are numeric, convert to float32
                A = np.asarray(A, dtype=np.float32)
            except (ValueError, TypeError) as e:
                # Re-raise with proper message format
                if "argument must be" not in str(e):
                    raise TypeError(
                        "argument must be a string or a real number"
                    ) from None
                else:
                    raise

    if handle is None:
        handle = Handle()
    cdef device_resources *h = <device_resources*><size_t>handle.getHandle()

    A, _n_rows, _n_cols, _ = \
        input_to_cuml_array(A, order="C", check_dtype=np.float32,
                            convert_to_dtype=cp.float32)
    A_ptr = <uintptr_t>A.ptr

    cdef params config

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

    transform(
        deref(h), config,
        make_device_matrix_view[float, int, row_major](
            <float *>A_ptr, <int> A.shape[0], <int> A.shape[1]),
        make_device_matrix_view[float, int, col_major](
            <float *>eigenvectors_ptr, <int> A.shape[0], <int> n_components))

    return eigenvectors


class SpectralEmbedding(Base,
                        InteropMixin,
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
    affinity_matrix_ : None
        Not implemented in cuML. Always None to save memory.
        The affinity matrix is not stored after computing the embedding.
    n_neighbors_ : int
        Number of nearest neighbors effectively used.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during fit. Defined only when X has feature
        names that are all strings.

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
    _cpu_class_path = "sklearn.manifold.SpectralEmbedding"
    embedding_ = CumlArrayDescriptor()

    def __init__(self, n_components=2, random_state=None, n_neighbors=None,
                 handle=None, verbose=False, output_type=None):
        super().__init__(handle=handle, verbose=verbose, output_type=output_type)
        self.n_components = n_components
        self.random_state = random_state
        self.n_neighbors = n_neighbors

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + [
            "n_components",
            "random_state",
            "n_neighbors"
        ]

    @classmethod
    def _params_from_cpu(cls, model):
        """Get parameters to use to instantiate a GPU model from a CPU model.

        Parameters
        ----------
        model : sklearn.manifold.SpectralEmbedding
            The CPU model to get parameters from

        Returns
        -------
        dict
            Parameters to pass to the GPU model constructor
        """
        affinity = getattr(model, 'affinity', 'nearest_neighbors')
        if affinity != 'nearest_neighbors':
            raise UnsupportedOnGPU(
                f"`affinity={affinity!r}` is not supported. "
                "Only 'nearest_neighbors' affinity is currently supported."
            )




        params = {
            "n_components": model.n_components,
            "random_state": model.random_state,
            "n_neighbors": model.n_neighbors
        }
        return params

    def _attrs_to_cpu(self, model):
        """Get attributes to set on CPU model from GPU model.

        Override the base implementation to include feature_names_in_.
        """
        # Get base attributes (n_features_in_)
        out = super()._attrs_to_cpu(model)

        # Add embedding_ attribute
        if hasattr(self, 'embedding_'):
            # Convert to numpy if it's a cupy array
            embedding = self.embedding_
            if hasattr(embedding, '__cuda_array_interface__'):
                embedding = cp.asnumpy(embedding)
            out['embedding_'] = embedding

        # Add feature_names_in_ if it exists
        if hasattr(self, 'feature_names_in_'):
            out['feature_names_in_'] = self.feature_names_in_

        # Add n_neighbors_ if it exists
        if hasattr(self, 'n_neighbors_'):
            out['n_neighbors_'] = self.n_neighbors_

        # Add affinity_matrix_ (always None in cuML)
        if hasattr(self, 'affinity_matrix_'):
            out['affinity_matrix_'] = self.affinity_matrix_

        return out

    def fit_transform(self, X, y=None) -> CumlArray:
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

    def fit(self, X, y=None) -> "SpectralEmbedding":
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
        # Convert sparse input to dense (cuML doesn't support sparse directly)
        try:
            import scipy.sparse as sp
            if sp.issparse(X):
                # Convert sparse to dense automatically
                X = X.toarray()
        except ImportError:
            pass

        # Check for 1D input
        if hasattr(X, 'ndim'):
            if X.ndim == 1:
                raise ValueError(
                    "Expected 2D array, got 1D array instead:\n"
                    f"array={X}.\n"
                    "Reshape your data either using array.reshape(-1, 1) if "
                    "your data has a single feature or array.reshape(1, -1) "
                    "if it contains a single sample."
                )
        elif hasattr(X, 'shape') and len(X.shape) == 1:
            raise ValueError(
                "Expected 2D array, got 1D array instead:\n"
                f"array shape={X.shape}.\n"
                "Reshape your data either using array.reshape(-1, 1) if "
                "your data has a single feature or array.reshape(1, -1) "
                "if it contains a single sample."
            )

        # Check for complex data
        if hasattr(X, 'dtype'):
            if np.iscomplexobj(X):
                raise ValueError("Complex data not supported")

        # Check for empty data
        if hasattr(X, 'shape'):
            if len(X.shape) >= 2:
                if X.shape[0] == 0:
                    raise ValueError(
                        f"Found array with 0 sample(s) (shape={X.shape}) while a "
                        f"minimum of 1 is required."
                    )
                if X.shape[1] == 0:
                    raise ValueError(
                        f"Found array with 0 feature(s) (shape={X.shape}) while a "
                        f"minimum of 1 is required."
                    )
                # SpectralEmbedding requires at least 2 samples
                if X.shape[0] < 2:
                    raise ValueError(
                        f"SpectralEmbedding requires at least 2 samples, but got "
                        f"only {X.shape[0]} sample(s)."
                    )
                # SpectralEmbedding requires at least 2 features for meaningful embedding
                if X.shape[1] < 2:
                    raise ValueError(
                        f"SpectralEmbedding requires at least 2 features for meaningful "
                        f"results, but got only {X.shape[1]} feature(s)."
                    )

        # Check for NaN and Inf values
        if hasattr(X, 'dtype'):
            # Check if it's a numeric dtype (not object)
            if (X.dtype != object and
                    not (hasattr(X.dtype, 'name') and X.dtype.name == 'object')):
                # Convert to numpy array if needed for checking
                X_check = X
                if hasattr(X, 'to_numpy'):  # cudf DataFrame
                    X_check = X.to_numpy()
                elif hasattr(X, 'values'):  # pandas DataFrame
                    X_check = X.values
                elif hasattr(X, '__cuda_array_interface__'):  # cupy array
                    X_check = cp.asnumpy(X)

                # Check for NaN and Inf
                X_np = np.asarray(X_check)
                if np.any(np.isnan(X_np)):
                    raise ValueError(
                        "Input contains NaN"
                    )
                if np.any(np.isinf(X_np)):
                    raise ValueError(
                        "Input contains infinity or a value too large for dtype('float64')."
                    )

        # Check for object dtype and validate contents
        if hasattr(X, 'dtype'):
            if X.dtype == object or (hasattr(X.dtype, 'name') and X.dtype.name == 'object'):
                # Try to convert to numeric array
                try:
                    # First check if any element is not numeric
                    flat_X = np.asarray(X).ravel()
                    for elem in flat_X:
                        # Check if element is a dict, list, or other non-numeric type
                        if isinstance(elem, (dict, list, tuple, str)) or not np.isscalar(elem):
                            raise TypeError(
                                "argument must be a string or a real number, not "
                                f"'{type(elem).__name__}'"
                            )
                    # If all elements are numeric, convert to float32
                    X = np.asarray(X, dtype=np.float32)
                except (ValueError, TypeError) as e:
                    # Re-raise with proper message format
                    if "argument must be" not in str(e):
                        raise TypeError(
                            "argument must be a string or a real number"
                        ) from None
                    else:
                        raise

        # Store feature names if X is a pandas DataFrame
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns)

        # Store the number of features - must be done before calling spectral_embedding
        # Convert X to get its shape for setting attributes
        from cuml.common import input_to_cuml_array
        _X_arr, n_samples, n_features, _ = input_to_cuml_array(
            X, order="C", check_dtype=np.float32, convert_to_dtype=cp.float32
        )

        # Always set n_features_in_ for sklearn compatibility
        self.n_features_in_ = n_features

        # Store n_neighbors_ for sklearn compatibility
        self.n_neighbors_ = (
            self.n_neighbors
            if self.n_neighbors is not None
            else max(int(n_samples / 10), 1)
        )

        self.embedding_ = spectral_embedding(
            X,
            n_components=self.n_components,
            random_state=self.random_state,
            n_neighbors=self.n_neighbors,
            handle=self.handle
        )

        # Set affinity_matrix_ to None for sklearn compatibility
        # cuML doesn't store the affinity matrix to save memory
        self.affinity_matrix_ = None

        return self

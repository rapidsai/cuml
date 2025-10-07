#
# Copyright (c) 2019-2025, NVIDIA CORPORATION.
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
import cupyx.scipy.sparse
import numpy as np

import cuml.internals
from cuml.common import using_output_type
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.doc_utils import generate_docstring
from cuml.common.exceptions import NotFittedError
from cuml.common.sparse_utils import is_sparse
from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from cuml.internals.input_utils import input_to_cuml_array
from cuml.internals.interop import (
    InteropMixin,
    UnsupportedOnGPU,
    to_cpu,
    to_gpu,
)
from cuml.internals.mixins import FMajorInputTagMixin, SparseInputTagMixin
from cuml.prims.stats import cov

from libc.stdint cimport uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t

from cuml.decomposition.common cimport paramsPCA, solver


cdef extern from "cuml/decomposition/pca.hpp" namespace "ML" nogil:

    cdef void pcaFit(handle_t& handle,
                     float *input,
                     float *components,
                     float *explained_var,
                     float *explained_var_ratio,
                     float *singular_vals,
                     float *mu,
                     float *noise_vars,
                     const paramsPCA &prms) except +

    cdef void pcaFit(handle_t& handle,
                     double *input,
                     double *components,
                     double *explained_var,
                     double *explained_var_ratio,
                     double *singular_vals,
                     double *mu,
                     double *noise_vars,
                     const paramsPCA &prms) except +

    cdef void pcaInverseTransform(handle_t& handle,
                                  float *trans_input,
                                  float *components,
                                  float *singular_vals,
                                  float *mu,
                                  float *input,
                                  const paramsPCA &prms) except +

    cdef void pcaInverseTransform(handle_t& handle,
                                  double *trans_input,
                                  double *components,
                                  double *singular_vals,
                                  double *mu,
                                  double *input,
                                  const paramsPCA &prms) except +

    cdef void pcaTransform(handle_t& handle,
                           float *input,
                           float *components,
                           float *trans_input,
                           float *singular_vals,
                           float *mu,
                           const paramsPCA &prms) except +

    cdef void pcaTransform(handle_t& handle,
                           double *input,
                           double *components,
                           double *trans_input,
                           double *singular_vals,
                           double *mu,
                           const paramsPCA &prms) except +


class PCA(Base,
          InteropMixin,
          FMajorInputTagMixin,
          SparseInputTagMixin):

    """
    PCA (Principal Component Analysis) is a fundamental dimensionality
    reduction technique used to combine features in X in linear combinations
    such that each new component captures the most information or variance of
    the data. N_components is usually small, say at 3, where it can be used for
    data visualization, data compression and exploratory analysis.

    cuML's PCA expects an array-like object or cuDF DataFrame, and provides 2
    algorithms Full and Jacobi. Full (default) uses a full eigendecomposition
    then selects the top K eigenvectors. The Jacobi algorithm is much faster
    as it iteratively tries to correct the top K eigenvectors, but might be
    less accurate.

    Examples
    --------

    .. code-block:: python

        >>> # Both import methods supported
        >>> from cuml import PCA
        >>> from cuml.decomposition import PCA

        >>> import cudf
        >>> import cupy as cp

        >>> gdf_float = cudf.DataFrame()
        >>> gdf_float['0'] = cp.asarray([1.0,2.0,5.0], dtype = cp.float32)
        >>> gdf_float['1'] = cp.asarray([4.0,2.0,1.0], dtype = cp.float32)
        >>> gdf_float['2'] = cp.asarray([4.0,2.0,1.0], dtype = cp.float32)

        >>> pca_float = PCA(n_components = 2)
        >>> pca_float.fit(gdf_float)
        PCA()

        >>> print(f'components: {pca_float.components_}') # doctest: +SKIP
        components: 0           1           2
        0  0.69225764  -0.5102837 -0.51028395
        1 -0.72165036 -0.48949987  -0.4895003
        >>> print(f'explained variance: {pca_float.explained_variance_}')
        explained variance: 0   8.510...
        1 0.489...
        dtype: float32
        >>> exp_var = pca_float.explained_variance_ratio_
        >>> print(f'explained variance ratio: {exp_var}')
        explained variance ratio: 0   0.9456...
        1 0.054...
        dtype: float32

        >>> print(f'singular values: {pca_float.singular_values_}')
        singular values: 0 4.125...
        1 0.989...
        dtype: float32
        >>> print(f'mean: {pca_float.mean_}')
        mean: 0 2.666...
        1 2.333...
        2 2.333...
        dtype: float32
        >>> trans_gdf_float = pca_float.transform(gdf_float)
        >>> print(f'Inverse: {trans_gdf_float}') # doctest: +SKIP
        Inverse: 0           1
        0   -2.8547091 -0.42891636
        1 -0.121316016  0.80743366
        2    2.9760244 -0.37851727
        >>> input_gdf_float = pca_float.inverse_transform(trans_gdf_float)
        >>> print(f'Input: {input_gdf_float}') # doctest: +SKIP
        Input: 0         1         2
        0 1.0 4.0 4.0
        1 2.0 2.0 2.0
        2 5.0 1.0 1.0

    Parameters
    ----------
    copy : boolean (default = True)
        If True, then copies data then removes mean from data. False might
        cause data to be overwritten with its mean centered version.
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    iterated_power : int (default = 15)
        Used in Jacobi solver. The more iterations, the more accurate, but
        slower.
    n_components : int (default = None)
        The number of top K singular vectors / values you want.
        Must be <= number(columns). If n_components is not set, then all
        components are kept:

            ``n_components = min(n_samples, n_features)``

    svd_solver : 'full' or 'jacobi' or 'auto' (default = 'full')
        Full uses a eigendecomposition of the covariance matrix then discards
        components.
        Jacobi is much faster as it iteratively corrects, but is less accurate.
    tol : float (default = 1e-7)
        Used if algorithm = "jacobi". Smaller tolerance can increase accuracy,
        but but will slow down the algorithm's convergence.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    whiten : boolean (default = False)
        If True, de-correlates the components. This is done by dividing them by
        the corresponding singular values then multiplying by sqrt(n_samples).
        Whitening allows each component to have unit variance and removes
        multi-collinearity. It might be beneficial for downstream
        tasks like LinearRegression where correlated features cause problems.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.

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

    components_ = CumlArrayDescriptor(order='F')
    explained_variance_ = CumlArrayDescriptor(order='F')
    explained_variance_ratio_ = CumlArrayDescriptor(order='F')
    singular_values_ = CumlArrayDescriptor(order='F')
    mean_ = CumlArrayDescriptor(order='F')

    _cpu_class_path = "sklearn.decomposition.PCA"

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + \
            ["copy", "iterated_power", "n_components", "svd_solver", "tol", "whiten"]

    @classmethod
    def _params_from_cpu(cls, model):
        if model.n_components == "mle":
            raise UnsupportedOnGPU("`n_components='mle'` is not supported")

        svd_solver = "auto" if model.svd_solver == "auto" else "full"

        # Since the solvers are different, we want to adjust the tolerances used.
        # TODO: here we only adjust the default tolerance, there's likely a better
        # conversion equation we should apply.
        if (tol := model.tol) == 0.0:
            tol = 1e-7

        if (iterated_power := model.iterated_power) == "auto":
            iterated_power = 15

        return {
            "n_components": model.n_components,
            "copy": model.copy,
            "whiten": model.whiten,
            "svd_solver": svd_solver,
            "tol": tol,
            "iterated_power": iterated_power,
        }

    def _params_to_cpu(self):
        # Since the solvers are different, we want to adjust the tolerances used.
        # TODO: here we only adjust the default tolerance, there's likely a better
        # conversion equation we should apply.
        if (tol := self.tol) == 1e-7:
            tol = 0.0

        svd_solver = "auto" if self.svd_solver == "jacobi" else self.svd_solver

        return {
            "n_components": self.n_components,
            "copy": self.copy,
            "whiten": self.whiten,
            "svd_solver": svd_solver,
            "tol": tol,
            "iterated_power": self.iterated_power,
        }

    def _attrs_from_cpu(self, model):
        return {
            "components_": to_gpu(model.components_, order="F"),
            "explained_variance_": to_gpu(model.explained_variance_, order="F"),
            "explained_variance_ratio_": to_gpu(model.explained_variance_ratio_, order="F"),
            "singular_values_": to_gpu(model.singular_values_, order="F"),
            "mean_": to_gpu(model.mean_, order="F"),
            "n_components_": model.n_components_,
            "n_samples_": model.n_samples_,
            "noise_variance_": model.noise_variance_,
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        return {
            "components_": to_cpu(self.components_),
            "explained_variance_": to_cpu(self.explained_variance_),
            "explained_variance_ratio_": to_cpu(self.explained_variance_ratio_),
            "singular_values_": to_cpu(self.singular_values_),
            "mean_": to_cpu(self.mean_),
            "n_components_": self.n_components_,
            "n_samples_": self.n_samples_,
            "noise_variance_": self.noise_variance_,
            **super()._attrs_to_cpu(model),
        }

    def __init__(self, *, copy=True, handle=None, iterated_power=15,
                 n_components=None, svd_solver='auto',
                 tol=1e-7, verbose=False, whiten=False,
                 output_type=None):
        super().__init__(handle=handle, verbose=verbose, output_type=output_type)
        self.copy = copy
        self.iterated_power = iterated_power
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.tol = tol
        self.whiten = whiten

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        # Exposed to support sklearn's `get_feature_names_out`
        return self.components_.shape[0]

    def _fit_dense(self, X):
        # Initialize parameters
        cdef paramsPCA params
        params.n_components = self.n_components_
        params.n_rows = self.n_samples_
        params.n_cols = self.n_features_in_
        params.whiten = self.whiten
        params.n_iterations = self.iterated_power
        params.tol = self.tol
        if self.svd_solver in ("auto", "full"):
            params.algorithm = solver.COV_EIG_DQ
        elif self.svd_solver == "jacobi":
            params.algorithm = solver.COV_EIG_JACOBI
        else:
            raise ValueError(
                f"Expected `svd_solver` to be one of ['auto', 'full', 'jacobi'], "
                f"got {self.svd_solver!r}"
            )

        # Allocate output arrays
        components = CumlArray.zeros(
            (self.n_components_, self.n_features_in_), dtype=X.dtype
        )
        explained_variance = CumlArray.zeros(self.n_components_, dtype=X.dtype)
        explained_variance_ratio = CumlArray.zeros(self.n_components_, dtype=X.dtype)
        mean = CumlArray.zeros(self.n_features_in_, dtype=X.dtype)
        singular_values = CumlArray.zeros(self.n_components_, dtype=X.dtype)
        noise_variance = CumlArray.zeros(1, dtype=X.dtype)

        cdef uintptr_t X_ptr = X.ptr
        cdef uintptr_t components_ptr = components.ptr
        cdef uintptr_t explained_variance_ptr = explained_variance.ptr
        cdef uintptr_t explained_variance_ratio_ptr = explained_variance_ratio.ptr
        cdef uintptr_t singular_values_ptr = singular_values.ptr
        cdef uintptr_t mean_ptr = mean.ptr
        cdef uintptr_t noise_variance_ptr = noise_variance.ptr
        cdef bool fit_float32 = (X.dtype == np.float32)
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        # Perform fit
        with nogil:
            if fit_float32:
                pcaFit(
                    handle_[0],
                    <float*> X_ptr,
                    <float*> components_ptr,
                    <float*> explained_variance_ptr,
                    <float*> explained_variance_ratio_ptr,
                    <float*> singular_values_ptr,
                    <float*> mean_ptr,
                    <float*> noise_variance_ptr,
                    params
                )
            else:
                pcaFit(
                    handle_[0],
                    <double*> X_ptr,
                    <double*> components_ptr,
                    <double*> explained_variance_ptr,
                    <double*> explained_variance_ratio_ptr,
                    <double*> singular_values_ptr,
                    <double*> mean_ptr,
                    <double*> noise_variance_ptr,
                    params
                )
        self.handle.sync()

        # Store results
        self.components_ = components
        self.explained_variance_ = explained_variance
        self.explained_variance_ratio_ = explained_variance_ratio
        self.mean_ = mean
        self.singular_values_ = singular_values
        self.noise_variance_ = float(noise_variance.to_output("numpy"))

    def _fit_sparse(self, X):
        covariance, mean, _ = cov(X, X, return_mean=True)

        explained_variance, components = cp.linalg.eigh(covariance, UPLO='U')

        # Reverse the eigen vector and eigen values here because cupy provides
        # them in ascending order.
        explained_variance = explained_variance[::-1]
        components = components[:, ::-1]

        if self.n_components_ < min(self.n_samples_, self.n_features_in_):
            noise_variance = float(explained_variance[self.n_components_:].mean())
        else:
            noise_variance = 0.0

        explained_variance_sum = explained_variance.sum()

        components = components.T[:self.n_components_, :]
        explained_variance = explained_variance[:self.n_components_]

        explained_variance_ratio = explained_variance / explained_variance_sum
        singular_values = cp.sqrt(
            cp.where(explained_variance < 0, 0, explained_variance) * (X.shape[0] - 1)
        )

        # Store results
        self.components_ = CumlArray(data=cp.asfortranarray(components))
        self.explained_variance_ = CumlArray(data=explained_variance)
        self.explained_variance_ratio_ = CumlArray(data=explained_variance_ratio)
        self.mean_ = CumlArray(data=mean.flatten())
        self.singular_values_ = CumlArray(data=singular_values)
        self.noise_variance_ = noise_variance

    @generate_docstring(X='dense_sparse')
    def fit(self, X, y=None, *, convert_dtype=True) -> "PCA":
        """
        Fit the model with X. y is currently ignored.

        """
        if (sparse := is_sparse(X)):
            X = cupyx.scipy.sparse.coo_matrix(X)
            n_rows, n_cols = X.shape
        else:
            X, n_rows, n_cols, _ = input_to_cuml_array(
                X,
                convert_to_dtype=(np.float32 if convert_dtype else None),
                check_dtype=[np.float32, np.float64],
            )

        self.n_samples_ = n_rows

        if self.n_components is None:
            self.n_components_ = min(n_rows, n_cols)
        elif self.n_components > n_cols:
            raise ValueError(
                f"`n_components` ({self.n_components}) must be <= than the "
                f"number of features in X ({n_cols})"
            )
        else:
            self.n_components_ = self.n_components

        if sparse:
            self._fit_sparse(X)
        else:
            self._fit_dense(X)
        return self

    @generate_docstring(X='dense_sparse',
                        return_values={'name': 'trans',
                                       'type': 'dense_sparse',
                                       'description': 'Transformed values',
                                       'shape': '(n_samples, n_components)'})
    @cuml.internals.api_base_return_array_skipall
    def fit_transform(self, X, y=None) -> CumlArray:
        """
        Fit the model with X and apply the dimensionality reduction on X.

        """
        return self.fit(X).transform(X)

    def _inverse_transform_sparse(self, X, return_sparse=False, sparse_tol=1e-10):
        X = cupyx.scipy.sparse.coo_matrix(X)

        with using_output_type("cupy"):
            components = self.components_
            explained_variance = self.explained_variance_
            mean = self.mean_

        if self.whiten:
            components = cp.sqrt(explained_variance[:, None]) * components

        out = X @ components
        out += mean

        if return_sparse:
            out[out < sparse_tol] = 0
            return cupyx.scipy.sparse.csr_matrix(out)

        return out

    def _inverse_transform_dense(self, X, convert_dtype=True):
        dtype = self.components_.dtype
        X_m, n_rows, _, _ = input_to_cuml_array(
            X,
            check_dtype=dtype,
            convert_to_dtype=(dtype if convert_dtype else None),
            check_cols=self.n_components_,
        )

        out = CumlArray.zeros((n_rows, self.n_features_in_), dtype=dtype)

        cdef paramsPCA params
        params.n_components = self.n_components_
        params.n_rows = n_rows
        params.n_cols = self.n_features_in_
        params.whiten = self.whiten

        cdef uintptr_t X_ptr = X_m.ptr
        cdef uintptr_t X_inv_ptr = out.ptr
        cdef uintptr_t components_ptr = self.components_.ptr
        cdef uintptr_t singular_values_ptr = self.singular_values_.ptr
        cdef uintptr_t mean_ptr = self.mean_.ptr
        cdef bool use_float32 = dtype == np.float32
        cdef handle_t* h_ = <handle_t*><size_t>self.handle.getHandle()

        with nogil:
            if use_float32:
                pcaInverseTransform(h_[0],
                                    <float*> X_ptr,
                                    <float*> components_ptr,
                                    <float*> singular_values_ptr,
                                    <float*> mean_ptr,
                                    <float*> X_inv_ptr,
                                    params)
            else:
                pcaInverseTransform(h_[0],
                                    <double*> X_ptr,
                                    <double*> components_ptr,
                                    <double*> singular_values_ptr,
                                    <double*> mean_ptr,
                                    <double*> X_inv_ptr,
                                    params)
        self.handle.sync()

        return out

    @generate_docstring(X='dense_sparse',
                        return_values={'name': 'X_inv',
                                       'type': 'dense_sparse',
                                       'description': 'Transformed values',
                                       'shape': '(n_samples, n_features)'})
    def inverse_transform(
        self,
        X,
        *,
        convert_dtype=False,
        return_sparse=False,
        sparse_tol=1e-10,
    ) -> CumlArray:
        """
        Transform data back to its original space.

        In other words, return an input X_original whose transform would be X.

        """
        self._check_is_fitted()
        if is_sparse(X):
            return self._inverse_transform_sparse(
                X, return_sparse=return_sparse, sparse_tol=sparse_tol
            )
        return self._inverse_transform_dense(X, convert_dtype=convert_dtype)

    def _transform_sparse(self, X):
        X = cupyx.scipy.sparse.coo_matrix(X)

        with using_output_type("cupy"):
            components = self.components_
            explained_variance = self.explained_variance_
            mean = self.mean_

        out = X @ components.T
        out -= (mean.reshape((1, -1)) @ components.T)
        if self.whiten:
            scale = cp.sqrt(explained_variance)
            min_scale = cp.finfo(scale.dtype).eps
            scale[scale < min_scale] = min_scale
            out /= scale
        return out

    def _transform_dense(self, X, convert_dtype=True):
        dtype = self.components_.dtype

        X_m, n_rows, n_cols, _ = input_to_cuml_array(
            X,
            check_dtype=dtype,
            convert_to_dtype=(dtype if convert_dtype else None),
            check_cols=self.n_features_in_,
        )

        out = CumlArray.zeros(
            (n_rows, self.n_components_), dtype=dtype, index=X_m.index
        )

        cdef paramsPCA params
        params.n_components = self.n_components_
        params.n_rows = n_rows
        params.n_cols = n_cols
        params.whiten = self.whiten

        cdef uintptr_t X_ptr = X_m.ptr
        cdef uintptr_t out_ptr = out.ptr
        cdef uintptr_t components_ptr = self.components_.ptr
        cdef uintptr_t singular_values_ptr = self.singular_values_.ptr
        cdef uintptr_t mean_ptr = self.mean_.ptr
        cdef bool use_float32 = dtype == np.float32
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        with nogil:
            if use_float32:
                pcaTransform(
                    handle_[0],
                    <float*> X_ptr,
                    <float*> components_ptr,
                    <float*> out_ptr,
                    <float*> singular_values_ptr,
                    <float*> mean_ptr,
                    params
                )
            else:
                pcaTransform(
                    handle_[0],
                    <double*> X_ptr,
                    <double*> components_ptr,
                    <double*> out_ptr,
                    <double*> singular_values_ptr,
                    <double*> mean_ptr,
                    params
                )
        self.handle.sync()
        return out

    @generate_docstring(X='dense_sparse',
                        return_values={'name': 'trans',
                                       'type': 'dense_sparse',
                                       'description': 'Transformed values',
                                       'shape': '(n_samples, n_components)'})
    def transform(self, X, *, convert_dtype=True) -> CumlArray:
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set.

        """
        self._check_is_fitted()

        if is_sparse(X):
            return self._transform_sparse(X)
        return self._transform_dense(X, convert_dtype=convert_dtype)

    def _check_is_fitted(self):
        if not hasattr(self, "components_"):
            msg = ("This instance is not fitted yet. Call 'fit' "
                   "with appropriate arguments before using this estimator.")
            raise NotFittedError(msg)

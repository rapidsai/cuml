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

import numpy as np

import cuml.internals
from cuml.common import input_to_cuml_array
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.doc_utils import generate_docstring
from cuml.internals.array import CumlArray
from cuml.internals.base import Base
from cuml.internals.interop import InteropMixin, to_cpu, to_gpu
from cuml.internals.mixins import FMajorInputTagMixin

from libc.stdint cimport uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t

from cuml.decomposition.common cimport paramsTSVD, solver


cdef extern from "cuml/decomposition/tsvd.hpp" namespace "ML" nogil:

    cdef void tsvdFitTransform(handle_t& handle,
                               float *input,
                               float *trans_input,
                               float *components,
                               float *explained_var,
                               float *explained_var_ratio,
                               float *singular_vals,
                               const paramsTSVD &prms) except +

    cdef void tsvdFitTransform(handle_t& handle,
                               double *input,
                               double *trans_input,
                               double *components,
                               double *explained_var,
                               double *explained_var_ratio,
                               double *singular_vals,
                               const paramsTSVD &prms) except +

    cdef void tsvdInverseTransform(handle_t& handle,
                                   float *trans_input,
                                   float *components,
                                   float *input,
                                   const paramsTSVD &prms) except +

    cdef void tsvdInverseTransform(handle_t& handle,
                                   double *trans_input,
                                   double *components,
                                   double *input,
                                   const paramsTSVD &prms) except +

    cdef void tsvdTransform(handle_t& handle,
                            float *input,
                            float *components,
                            float *trans_input,
                            const paramsTSVD &prms) except +

    cdef void tsvdTransform(handle_t& handle,
                            double *input,
                            double *components,
                            double *trans_input,
                            const paramsTSVD &prms) except +


class TruncatedSVD(Base,
                   InteropMixin,
                   FMajorInputTagMixin):
    """
    TruncatedSVD is used to compute the top K singular values and vectors of a
    large matrix X. It is much faster when n_components is small, such as in
    the use of PCA when 3 components is used for 3D visualization.

    cuML's TruncatedSVD an array-like object or cuDF DataFrame, and provides 2
    algorithms Full and Jacobi. Full (default) uses a full eigendecomposition
    then selects the top K singular vectors. The Jacobi algorithm is much
    faster as it iteratively tries to correct the top K singular vectors, but
    might be less accurate.

    Examples
    --------

    .. code-block:: python

        >>> # Both import methods supported
        >>> from cuml import TruncatedSVD
        >>> from cuml.decomposition import TruncatedSVD

        >>> import cudf
        >>> import cupy as cp

        >>> gdf_float = cudf.DataFrame()
        >>> gdf_float['0'] = cp.asarray([1.0,2.0,5.0], dtype=cp.float32)
        >>> gdf_float['1'] = cp.asarray([4.0,2.0,1.0], dtype=cp.float32)
        >>> gdf_float['2'] = cp.asarray([4.0,2.0,1.0], dtype=cp.float32)

        >>> tsvd_float = TruncatedSVD(n_components = 2, algorithm = "jacobi",
        ...                           n_iter = 20, tol = 1e-9)
        >>> tsvd_float.fit(gdf_float)
        TruncatedSVD()
        >>> print(f'components: {tsvd_float.components_}') # doctest: +SKIP
        components:           0         1         2
        0  0.587259  0.572331  0.572331
        1  0.809399 -0.415255 -0.415255
        >>> exp_var = tsvd_float.explained_variance_
        >>> print(f'explained variance: {exp_var}')
        explained variance: 0    0.494...
        1    5.505...
        dtype: float32
        >>> exp_var_ratio = tsvd_float.explained_variance_ratio_
        >>> print(f'explained variance ratio: {exp_var_ratio}')
        explained variance ratio: 0    0.082...
        1    0.917...
        dtype: float32
        >>> sing_values = tsvd_float.singular_values_
        >>> print(f'singular values: {sing_values}')
        singular values: 0    7.439...
        1    4.081...
        dtype: float32

        >>> trans_gdf_float = tsvd_float.transform(gdf_float)
        >>> print(f'Transformed matrix: {trans_gdf_float}') # doctest: +SKIP
        Transformed matrix:           0         1
        0  5.165910 -2.512643
        1  3.463844 -0.042223
        2  4.080960  3.216484
        >>> input_gdf_float = tsvd_float.inverse_transform(trans_gdf_float)
        >>> print(f'Input matrix: {input_gdf_float}')
        Input matrix:      0    1    2
        0  1.0  4.0  4.0
        1  2.0  2.0  2.0
        2  5.0  1.0  1.0

    Parameters
    ----------
    algorithm : 'full' or 'jacobi' or 'auto' (default = 'full')
        Full uses a eigendecomposition of the covariance matrix then discards
        components.
        Jacobi is much faster as it iteratively corrects, but is less accurate.
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
    n_iter : int (default = 15)
        Used in Jacobi solver. The more iterations, the more accurate, but
        slower.
    random_state : int / None (default = None)
        If you want results to be the same when you restart Python, select a
        state.
    tol : float (default = 1e-7)
        Used if algorithm = "jacobi". Smaller tolerance can increase accuracy,
        but but will slow down the algorithm's convergence.
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
    components_ : array
        The top K components (VT.T[:,:n_components]) in U, S, VT = svd(X)
    explained_variance_ : array
        How much each component explains the variance in the data given by S**2
    explained_variance_ratio_ : array
        How much in % the variance is explained given by S**2/sum(S**2)
    singular_values_ : array
        The top K singular values. Remember all singular values >= 0

    Notes
    -----
    TruncatedSVD (the randomized version [Jacobi]) is fantastic when the number
    of components you want is much smaller than the number of features. The
    approximation to the largest singular values and vectors is very robust,
    however, this method loses a lot of accuracy when you want many, many
    components.

    **Applications of TruncatedSVD**

    TruncatedSVD is also known as Latent Semantic Indexing (LSI) which
    tries to find topics of a word count matrix. If X previously was
    centered with mean removal, TruncatedSVD is the same as TruncatedPCA.
    TruncatedSVD is also used in information retrieval tasks,
    recommendation systems and data compression.

    For additional documentation, see `scikitlearn's TruncatedSVD docs
    <http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html>`_.

    """

    components_ = CumlArrayDescriptor(order='F')
    explained_variance_ = CumlArrayDescriptor(order='F')
    explained_variance_ratio_ = CumlArrayDescriptor(order='F')
    singular_values_ = CumlArrayDescriptor(order='F')

    _cpu_class_path = "sklearn.decomposition.TruncatedSVD"

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + \
            ["algorithm", "n_components", "n_iter", "random_state", "tol"]

    @classmethod
    def _params_from_cpu(cls, model):
        # Since the solvers are different, we want to adjust tol & n_iter.
        # TODO: here we only adjust the default values, there's likely a better
        # conversion equation we should apply.
        if (tol := model.tol) == 0.0:
            tol = 1e-7
        if (n_iter := model.n_iter) == 5:
            n_iter = 15

        return {
            "n_components": model.n_components,
            "algorithm": "full",
            "n_iter": n_iter,
            "tol": tol,
        }

    def _params_to_cpu(self):
        # Since the solvers are different, we want to adjust tol & n_iter.
        # TODO: here we only adjust the default values, there's likely a better
        # conversion equation we should apply.
        if (tol := self.tol) == 1e-7:
            tol = 0.0
        if (n_iter := self.n_iter) == 15:
            n_iter = 5

        return {
            "n_components": self.n_components,
            "algorithm": "randomized",
            "n_iter": n_iter,
            "tol": tol,
        }

    def _attrs_from_cpu(self, model):
        return {
            "components_": to_gpu(model.components_, order="F"),
            "explained_variance_": to_gpu(model.explained_variance_, order="F"),
            "explained_variance_ratio_": to_gpu(model.explained_variance_ratio_, order="F"),
            "singular_values_": to_gpu(model.singular_values_, order="F"),
            **super()._attrs_from_cpu(model),
        }

    def _attrs_to_cpu(self, model):
        return {
            "components_": to_cpu(self.components_),
            "explained_variance_": to_cpu(self.explained_variance_),
            "explained_variance_ratio_": to_cpu(self.explained_variance_ratio_),
            "singular_values_": to_cpu(self.singular_values_),
            **super()._attrs_to_cpu(model),
        }

    def __init__(self, *, algorithm='full', handle=None, n_components=1,
                 n_iter=15, random_state=None, tol=1e-7,
                 verbose=False, output_type=None):
        super().__init__(handle=handle, verbose=verbose, output_type=output_type)
        self.algorithm = algorithm
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        self.tol = tol

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        # Exposed to support sklearn's `get_feature_names_out`
        return self.components_.shape[0]

    @generate_docstring()
    def fit(self, X, y=None) -> "TruncatedSVD":
        """
        Fit model on training cudf DataFrame X. y is currently ignored.

        """
        self.fit_transform(X)
        return self

    @generate_docstring(return_values={'name': 'trans',
                                       'type': 'dense',
                                       'description': 'Reduced version of X',
                                       'shape': '(n_samples, n_components)'})
    @cuml.internals.api_base_fit_transform()
    def fit_transform(self, X, y=None, *, convert_dtype=True) -> CumlArray:
        """
        Fit model to X and perform dimensionality reduction on X.
        y is currently ignored.

        """
        # Validate input
        X_m, n_rows, n_cols, dtype = input_to_cuml_array(
            X,
            convert_to_dtype=(np.float32 if convert_dtype else None),
            check_dtype=[np.float32, np.float64]
        )

        # Validate and initialize parameters
        if self.n_components > n_cols:
            raise ValueError(
                f"`n_components` ({self.n_components}) must be <= than the "
                f"number of features in X ({n_cols})"
            )

        cdef paramsTSVD params
        params.n_components = self.n_components
        params.n_rows = n_rows
        params.n_cols = n_cols
        params.n_iterations = self.n_iter
        params.tol = self.tol
        if self.algorithm in ("auto", "full"):
            params.algorithm = solver.COV_EIG_DQ
        elif self.algorithm == "jacobi":
            params.algorithm = solver.COV_EIG_JACOBI
        else:
            raise ValueError(
                f"Expected `algorithm` to be one of ['auto', 'full', 'jacobi'], "
                f"got {self.algorithm!r}"
            )

        # Allocate output arrays
        components = CumlArray.zeros((self.n_components, n_cols), dtype=dtype)
        explained_variance = CumlArray.zeros(self.n_components, dtype=dtype)
        explained_variance_ratio = CumlArray.zeros(self.n_components, dtype=dtype)
        singular_values = CumlArray.zeros(self.n_components, dtype=dtype)
        out = CumlArray.zeros((n_rows, self.n_components), dtype=dtype, index=X_m.index)

        cdef uintptr_t X_ptr = X_m.ptr
        cdef uintptr_t components_ptr = components.ptr
        cdef uintptr_t explained_variance_ptr = explained_variance.ptr
        cdef uintptr_t explained_variance_ratio_ptr = explained_variance_ratio.ptr
        cdef uintptr_t singular_values_ptr = singular_values.ptr
        cdef uintptr_t out_ptr = out.ptr
        cdef bool use_float32 = dtype == np.float32
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        # Perform fit
        with nogil:
            if use_float32:
                tsvdFitTransform(
                    handle_[0],
                    <float*> X_ptr,
                    <float*> out_ptr,
                    <float*> components_ptr,
                    <float*> explained_variance_ptr,
                    <float*> explained_variance_ratio_ptr,
                    <float*> singular_values_ptr,
                    params
                )
            else:
                tsvdFitTransform(
                    handle_[0],
                    <double*> X_ptr,
                    <double*> out_ptr,
                    <double*> components_ptr,
                    <double*> explained_variance_ptr,
                    <double*> explained_variance_ratio_ptr,
                    <double*> singular_values_ptr,
                    params
                )
        self.handle.sync()

        # Store results
        self.components_ = components
        self.explained_variance_ = explained_variance
        self.explained_variance_ratio_ = explained_variance_ratio
        self.singular_values_ = singular_values

        return out

    @generate_docstring(return_values={'name': 'X_original',
                                       'type': 'dense',
                                       'description': 'X in original space',
                                       'shape': '(n_samples, n_features)'})
    def inverse_transform(self, X, *, convert_dtype=False) -> CumlArray:
        """
        Transform X back to its original space.
        Returns X_original whose transform would be X.

        """
        dtype = self.components_.dtype
        X_m, n_rows, _, _ = input_to_cuml_array(
            X,
            check_dtype=dtype,
            convert_to_dtype=(dtype if convert_dtype else None),
            check_cols=self.n_components,
        )

        cdef paramsTSVD params
        params.n_components = self.n_components
        params.n_rows = n_rows
        params.n_cols = self.n_features_in_

        out = CumlArray.zeros(
            (n_rows, self.n_features_in_), dtype=dtype, index=X_m.index
        )

        cdef uintptr_t X_ptr = X_m.ptr
        cdef uintptr_t out_ptr = out.ptr
        cdef uintptr_t components_ptr = self.components_.ptr
        cdef bool use_float32 = dtype == np.float32
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        with nogil:
            if use_float32:
                tsvdInverseTransform(
                    handle_[0],
                    <float*> X_ptr,
                    <float*> components_ptr,
                    <float*> out_ptr,
                    params
                )
            else:
                tsvdInverseTransform(
                    handle_[0],
                    <double*> X_ptr,
                    <double*> components_ptr,
                    <double*> out_ptr,
                    params
                )
        self.handle.sync()

        return out

    @generate_docstring(return_values={'name': 'X_new',
                                       'type': 'dense',
                                       'description': 'Reduced version of X',
                                       'shape': '(n_samples, n_components)'})
    def transform(self, X, *, convert_dtype=True) -> CumlArray:
        """
        Perform dimensionality reduction on X.

        """
        dtype = self.components_.dtype
        X_m, n_rows, _, _ = input_to_cuml_array(
            X,
            check_dtype=dtype,
            convert_to_dtype=(dtype if convert_dtype else None),
            check_cols=self.n_features_in_,
        )

        cdef paramsTSVD params
        params.n_components = self.n_components
        params.n_rows = n_rows
        params.n_cols = self.n_features_in_

        out = CumlArray.zeros((n_rows, self.n_components), dtype=dtype, index=X_m.index)

        cdef uintptr_t X_ptr = X_m.ptr
        cdef uintptr_t out_ptr = out.ptr
        cdef uintptr_t components_ptr = self.components_.ptr
        cdef bool use_float32 = dtype == np.float32
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        with nogil:
            if use_float32:
                tsvdTransform(
                    handle_[0],
                    <float*> X_ptr,
                    <float*> components_ptr,
                    <float*> out_ptr,
                    params
                )
            else:
                tsvdTransform(
                    handle_[0],
                    <double*> X_ptr,
                    <double*> components_ptr,
                    <double*> out_ptr,
                    params
                )
        self.handle.sync()

        return out

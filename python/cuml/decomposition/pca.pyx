#
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

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import ctypes
import cudf
import numpy as np

from numba import cuda

from libcpp cimport bool
from libc.stdint cimport uintptr_t

from cuml.common.base import Base
from cuml.common.handle cimport cumlHandle
from cuml.decomposition.utils cimport *


cdef extern from "pca/pca.hpp" namespace "ML":

    cdef void pcaFit(cumlHandle& handle,
                     float *input,
                     float *components,
                     float *explained_var,
                     float *explained_var_ratio,
                     float *singular_vals,
                     float *mu,
                     float *noise_vars,
                     paramsPCA prms)

    cdef void pcaFit(cumlHandle& handle,
                     double *input,
                     double *components,
                     double *explained_var,
                     double *explained_var_ratio,
                     double *singular_vals,
                     double *mu,
                     double *noise_vars,
                     paramsPCA prms)

    cdef void pcaFitTransform(cumlHandle& handle,
                              float *input,
                              float *trans_input,
                              float *components,
                              float *explained_var,
                              float *explained_var_ratio,
                              float *singular_vals,
                              float *mu,
                              float *noise_vars,
                              paramsPCA prms)

    cdef void pcaFitTransform(cumlHandle& handle,
                              double *input,
                              double *trans_input,
                              double *components,
                              double *explained_var,
                              double *explained_var_ratio,
                              double *singular_vals,
                              double *mu,
                              double *noise_vars,
                              paramsPCA prms)

    cdef void pcaInverseTransform(cumlHandle& handle,
                                  float *trans_input,
                                  float *components,
                                  float *singular_vals,
                                  float *mu,
                                  float *input,
                                  paramsPCA prms)

    cdef void pcaInverseTransform(cumlHandle& handle,
                                  double *trans_input,
                                  double *components,
                                  double *singular_vals,
                                  double *mu,
                                  double *input,
                                  paramsPCA prms)

    cdef void pcaTransform(cumlHandle& handle,
                           float *input,
                           float *components,
                           float *trans_input,
                           float *singular_vals,
                           float *mu,
                           paramsPCA prms)

    cdef void pcaTransform(cumlHandle& handle,
                           double *input,
                           double *components,
                           double *trans_input,
                           double *singular_vals,
                           double *mu,
                           paramsPCA prms)


class PCA(Base):
    """
    PCA (Principal Component Analysis) is a fundamental dimensionality
    reduction technique used to combine features in X in linear combinations
    such that each new component captures the most information or variance of
    the data. N_components is usually small, say at 3, where it can be used for
    data visualization, data compression and exploratory analysis.

    cuML's PCA expects a cuDF DataFrame, and provides 2 algorithms Full and
    Jacobi. Full (default) uses a full eigendecomposition then selects the top
    K eigenvectors. The Jacobi algorithm is much faster as it iteratively tries
    to correct the top K eigenvectors, but might be less accurate.

    Examples
    ---------

    .. code-block:: python

        # Both import methods supported
        from cuml import PCA
        from cuml.decomposition import PCA

        import cudf
        import numpy as np

        gdf_float = cudf.DataFrame()
        gdf_float['0'] = np.asarray([1.0,2.0,5.0], dtype = np.float32)
        gdf_float['1'] = np.asarray([4.0,2.0,1.0], dtype = np.float32)
        gdf_float['2'] = np.asarray([4.0,2.0,1.0], dtype = np.float32)

        pca_float = PCA(n_components = 2)
        pca_float.fit(gdf_float)

        print(f'components: {pca_float.components_}')
        print(f'explained variance: {pca_float.explained_variance_}')
        exp_var = pca_float.explained_variance_ratio_
        print(f'explained variance ratio: {exp_var}')

        print(f'singular values: {pca_float.singular_values_}')
        print(f'mean: {pca_float.mean_}')
        print(f'noise variance: {pca_float.noise_variance_}')

        trans_gdf_float = pca_float.transform(gdf_float)
        print(f'Inverse: {trans_gdf_float}')

        input_gdf_float = pca_float.inverse_transform(trans_gdf_float)
        print(f'Input: {input_gdf_float}')

    Output:

    .. code-block:: python

          components:
                      0           1           2
                      0  0.69225764  -0.5102837 -0.51028395
                      1 -0.72165036 -0.48949987  -0.4895003

          explained variance:

                      0   8.510402
                      1 0.48959687

          explained variance ratio:

                       0   0.9456003
                       1 0.054399658

          singular values:

                     0 4.1256275
                     1 0.9895422

          mean:

                    0 2.6666667
                    1 2.3333333
                    2 2.3333333

          noise variance:

                0  0.0

          transformed matrix:
                       0           1
                       0   -2.8547091 -0.42891636
                       1 -0.121316016  0.80743366
                       2    2.9760244 -0.37851727

          Input Matrix:
                    0         1         2
                    0 1.0000001 3.9999993       4.0
                    1       2.0 2.0000002 1.9999999
                    2 4.9999995 1.0000006       1.0

    Parameters
    ----------
    copy : boolean (default = True)
        If True, then copies data then removes mean from data. False might
        cause data to be overwritten with its mean centered version.
    handle : cuml.Handle
        If it is None, a new one is created just for this class
    iterated_power : int (default = 15)
        Used in Jacobi solver. The more iterations, the more accurate, but
        slower.
    n_components : int (default = 1)
        The number of top K singular vectors / values you want.
        Must be <= number(columns).
    random_state : int / None (default = None)
        If you want results to be the same when you restart Python, select a
        state.
    svd_solver : 'full' or 'jacobi' or 'auto' (default = 'full')
        Full uses a eigendecomposition of the covariance matrix then discards
        components.
        Jacobi is much faster as it iteratively corrects, but is less accurate.
    tol : float (default = 1e-7)
        Used if algorithm = "jacobi". Smaller tolerance can increase accuracy,
        but but will slow down the algorithm's convergence.
    verbose : bool
        Whether to print debug spews
    whiten : boolean (default = False)
        If True, de-correlates the components. This is done by dividing them by
        the corresponding singular values then multiplying by sqrt(n_samples).
        Whitening allows each component to have unit variance and removes
        multi-collinearity. It might be beneficial for downstream
        tasks like LinearRegression where correlated features cause problems.


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
    ------
    PCA considers linear combinations of features, specifically those that
    maximise global variance structure. This means PCA is fantastic for global
    structure analyses, but weak for local relationships. Consider UMAP or
    T-SNE for a locally important embedding.

    **Applications of PCA**

        PCA is used extensively in practice for data visualization and data
        compression. It has been used to visualize extremely large word
        embeddings like Word2Vec and GloVe in 2 or 3 dimensions, large
        datasets of everyday objects and images, and used to distinguish
        between cancerous cells from healthy cells.


    For an additional example see `the PCA notebook
    <https://github.com/rapidsai/notebooks/blob/master/cuml/pca_demo.ipynb>`_.
    For additional docs, see `scikitlearn's PCA
    <http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>`_.
    """

    def __init__(self, copy=True, handle=None, iterated_power=15,
                 n_components=1, random_state=None, svd_solver='auto',
                 tol=1e-7, verbose=False, whiten=False):
        # parameters
        super(PCA, self).__init__(handle=handle, verbose=verbose)
        self.copy = copy
        self.iterated_power = iterated_power
        self.n_components = n_components
        self.random_state = random_state
        self.svd_solver = svd_solver
        self.tol = tol
        self.whiten = whiten
        self.c_algorithm = self._get_algorithm_c_name(self.svd_solver)
        # attributes
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.noise_variance_ = None
        self.components_ptr = None
        self.explained_variance_ptr = None
        self.explained_variance_ratio_ptr = None
        self.singular_values_ptr = None
        self.mean_ptr = None
        self.noise_variance_ptr = None

    def _get_algorithm_c_name(self, algorithm):
        algo_map = {
            'full': COV_EIG_DQ,
            'auto': COV_EIG_DQ,
            # 'arpack': NOT_SUPPORTED,
            # 'randomized': NOT_SUPPORTED,
            'jacobi': COV_EIG_JACOBI
        }
        if algorithm not in algo_map:
            msg = "algorithm {!r} is not supported"
            raise TypeError(msg.format(algorithm))
        return algo_map[algorithm]

    def _initialize_arrays(self, n_components, n_rows, n_cols):

        self.trans_input_ = cuda.to_device(np.zeros(n_rows*n_components,
                                                    dtype=self.gdf_datatype))
        self.components_ = cuda.to_device(np.zeros(n_components*n_cols,
                                                   dtype=self.gdf_datatype))
        self.explained_variance_ = cudf.Series(np.zeros(n_components,
                                               dtype=self.gdf_datatype))
        self.explained_variance_ratio_ = cudf.Series(np.zeros(n_components,
                                                     dtype=self.gdf_datatype))
        self.mean_ = cudf.Series(np.zeros(n_cols, dtype=self.gdf_datatype))
        self.singular_values_ = cudf.Series(np.zeros(n_components,
                                                     dtype=self.gdf_datatype))
        self.noise_variance_ = cudf.Series(np.zeros(1,
                                                    dtype=self.gdf_datatype))

    def fit(self, X, _transform=False):
        """
        Fit the model with X.

        Parameters
        ----------
        X : cuDF DataFrame
          Dense matrix (floats or doubles) of shape (n_samples, n_features)

        Returns
        -------
        cluster labels

        """
        # c params

        cdef uintptr_t input_ptr
        if (isinstance(X, cudf.DataFrame)):
            self.gdf_datatype = np.dtype(X[X.columns[0]]._column.dtype)
            # PCA expects transpose of the input
            X_m = X.as_gpu_matrix()
            self.n_rows = len(X)
            self.n_cols = len(X._cols)
        elif (isinstance(X, np.ndarray)):
            self.gdf_datatype = X.dtype
            X_m = cuda.to_device(np.array(X, order='F'))
            self.n_rows = X.shape[0]
            self.n_cols = X.shape[1]
        else:
            msg = "X matrix format  not supported"
            raise TypeError(msg)

        input_ptr = self._get_dev_array_ptr(X_m)

        cpdef paramsPCA params
        params.n_components = self.n_components
        params.n_rows = self.n_rows
        params.n_cols = self.n_cols
        params.whiten = self.whiten
        params.n_iterations = self.iterated_power
        params.tol = self.tol
        params.algorithm = self.c_algorithm

        if self.n_components > self.n_cols:
            raise ValueError('Number of components should not be greater than'
                             'the number of columns in the data')

        self._initialize_arrays(params.n_components,
                                params.n_rows, params.n_cols)

        cdef uintptr_t comp_ptr = self._get_dev_array_ptr(self.components_)

        cdef uintptr_t explained_var_ptr = \
            self._get_cudf_column_ptr(self.explained_variance_)

        cdef uintptr_t explained_var_ratio_ptr = \
            self._get_cudf_column_ptr(self.explained_variance_ratio_)

        cdef uintptr_t singular_vals_ptr = \
            self._get_cudf_column_ptr(self.singular_values_)

        cdef uintptr_t mean_ptr = self._get_cudf_column_ptr(self.mean_)

        cdef uintptr_t noise_vars_ptr = \
            self._get_cudf_column_ptr(self.noise_variance_)

        cdef uintptr_t t_input_ptr = self._get_dev_array_ptr(self.trans_input_)

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()
        if self.gdf_datatype.type == np.float32:
            pcaFitTransform(handle_[0],
                            <float*> input_ptr,
                            <float*> t_input_ptr,
                            <float*> comp_ptr,
                            <float*> explained_var_ptr,
                            <float*> explained_var_ratio_ptr,
                            <float*> singular_vals_ptr,
                            <float*> mean_ptr,
                            <float*> noise_vars_ptr,
                            params)
        else:
            pcaFitTransform(handle_[0],
                            <double*> input_ptr,
                            <double*> t_input_ptr,
                            <double*> comp_ptr,
                            <double*> explained_var_ptr,
                            <double*> explained_var_ratio_ptr,
                            <double*> singular_vals_ptr,
                            <double*> mean_ptr,
                            <double*> noise_vars_ptr,
                            params)

        # make sure the previously scheduled gpu tasks are complete before the
        # following transfers start
        self.handle.sync()

        components_gdf = cudf.DataFrame()
        for i in range(0, params.n_cols):
            n_c = params.n_components
            components_gdf[str(i)] = self.components_[i*n_c:(i+1)*n_c]

        self.components_ = components_gdf
        self.components_ptr = comp_ptr
        self.explained_variance_ptr = explained_var_ptr
        self.explained_variance_ratio_ptr = explained_var_ratio_ptr
        self.singular_values_ptr = singular_vals_ptr
        self.mean_ptr = mean_ptr
        self.noise_variance_ptr = noise_vars_ptr

        if (isinstance(X, cudf.DataFrame)):
            del(X_m)

        if not _transform:
            del(self.trans_input_)

        return self

    def fit_transform(self, X, y=None):
        """
        Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : cuDF DataFrame, shape (n_samples, n_features)
          training data (floats or doubles), where n_samples is the number of
          samples, and n_features is the number of features.

        y : ignored

        Returns
        -------
        X_new : cuDF DataFrame, shape (n_samples, n_components)
        """
        self.fit(X, _transform=True)
        X_new = cudf.DataFrame()
        num_rows = self.n_rows

        for i in range(0, self.n_components):
            X_new[str(i)] = self.trans_input_[i*num_rows:(i+1)*num_rows]

        return X_new

    def inverse_transform(self, X):
        """
        Transform data back to its original space.

        In other words, return an input X_original whose transform would be X.

        Parameters
        ----------
        X : cuDF DataFrame, shape (n_samples, n_components)
            New data (floats or doubles), where n_samples is the number of
            samples and n_components is the number of components.

        Returns
        -------
        X_original : cuDF DataFrame, shape (n_samples, n_features)

        """
        cdef uintptr_t trans_input_ptr
        if (isinstance(X, cudf.DataFrame)):
            gdf_datatype = np.dtype(X[X.columns[0]]._column.dtype)
            X_m = X.as_gpu_matrix()
        elif (isinstance(X, np.ndarray)):
            gdf_datatype = X.dtype
            X_m = cuda.to_device(np.array(X, order='F'))
        else:
            msg = "X matrix format  not supported"
            raise TypeError(msg)

        trans_input_ptr = self._get_dev_array_ptr(X_m)

        cpdef paramsPCA params
        params.n_components = self.n_components
        params.n_rows = len(X)
        params.n_cols = self.n_cols
        params.whiten = self.whiten

        input_data = cuda.to_device(np.zeros(params.n_rows*params.n_cols,
                                             dtype=gdf_datatype.type))

        cdef uintptr_t input_ptr = input_data.device_ctypes_pointer.value

        cdef uintptr_t components_ptr = self.components_ptr
        cdef uintptr_t singular_vals_ptr = self.singular_values_ptr
        cdef uintptr_t mean_ptr = self.mean_ptr

        cdef cumlHandle* h_ = <cumlHandle*><size_t>self.handle.getHandle()
        if gdf_datatype.type == np.float32:
            pcaInverseTransform(h_[0],
                                <float*> trans_input_ptr,
                                <float*> components_ptr,
                                <float*> singular_vals_ptr,
                                <float*> mean_ptr,
                                <float*> input_ptr,
                                params)
        else:
            pcaInverseTransform(h_[0],
                                <double*> trans_input_ptr,
                                <double*> components_ptr,
                                <double*> singular_vals_ptr,
                                <double*> mean_ptr,
                                <double*> input_ptr,
                                params)

        # make sure the previously scheduled gpu tasks are complete before the
        # following transfers start
        self.handle.sync()

        X_original = cudf.DataFrame()
        for i in range(0, params.n_cols):
            n_r = params.n_rows
            X_original[str(i)] = input_data[i*n_r:(i+1)*n_r]

        del(X_m)

        return X_original

    def transform(self, X):
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set.

        Parameters
        ----------
        X : cuDF DataFrame, shape (n_samples, n_features)
            New data (floats or doubles), where n_samples is the number of
            samples and n_features is the number of features.

        Returns
        -------
        X_new : cuDF DataFrame, shape (n_samples, n_components)

        """

        cdef uintptr_t input_ptr
        if (isinstance(X, cudf.DataFrame)):
            gdf_datatype = np.dtype(X[X.columns[0]]._column.dtype)
            X_m = X.as_gpu_matrix()
            n_rows = len(X)
            n_cols = len(X._cols)

        elif (isinstance(X, np.ndarray)):
            gdf_datatype = X.dtype
            X_m = cuda.to_device(np.array(X, order='F'))
            n_rows = X.shape[0]
            n_cols = X.shape[1]

        else:
            msg = "X matrix format  not supported"
            raise TypeError(msg)

        input_ptr = self._get_dev_array_ptr(X_m)

        cpdef paramsPCA params
        params.n_components = self.n_components
        params.n_rows = n_rows
        params.n_cols = n_cols
        params.whiten = self.whiten

        t_input_data = \
            cuda.to_device(np.zeros(params.n_rows*params.n_components,
                                    dtype=gdf_datatype.type))

        cdef uintptr_t trans_input_ptr = self._get_dev_array_ptr(t_input_data)
        cdef uintptr_t components_ptr = self.components_ptr
        cdef uintptr_t singular_vals_ptr = self.singular_values_ptr
        cdef uintptr_t mean_ptr = self.mean_ptr

        cdef cumlHandle* handle_ = <cumlHandle*><size_t>self.handle.getHandle()
        if gdf_datatype.type == np.float32:
            pcaTransform(handle_[0],
                         <float*> input_ptr,
                         <float*> components_ptr,
                         <float*> trans_input_ptr,
                         <float*> singular_vals_ptr,
                         <float*> mean_ptr,
                         params)
        else:
            pcaTransform(handle_[0],
                         <double*> input_ptr,
                         <double*> components_ptr,
                         <double*> trans_input_ptr,
                         <double*> singular_vals_ptr,
                         <double*> mean_ptr,
                         params)

        # make sure the previously scheduled gpu tasks are complete before the
        # following transfers start
        self.handle.sync()

        X_new = cudf.DataFrame()
        for i in range(0, params.n_components):
            X_new[str(i)] = t_input_data[i*params.n_rows:(i+1)*params.n_rows]

        del(X_m)
        return X_new

    def get_param_names(self):
        return ["copy", "iterated_power", "n_components", "svd_solver", "tol",
                "whiten"]

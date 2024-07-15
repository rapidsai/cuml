# distutils: language = c++

from cuml.internals.safe_imports import cpu_only_import
np = cpu_only_import('numpy')
from cuml.internals.safe_imports import gpu_only_import
cp = gpu_only_import('cupy')
cupyx = gpu_only_import('cupyx')
scipy = cpu_only_import('scipy')

rmm = gpu_only_import('rmm')

from libc.stdint cimport uintptr_t

import cuml.internals
from cuml.internals.array import CumlArray
from cuml.internals.base import UniversalBase
from cuml.common.doc_utils import generate_docstring
from cuml.internals.input_utils import input_to_cuml_array
from cuml.internals.input_utils import input_to_cupy_array
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common import using_output_type
from cuml.prims.stats import cov
from cuml.internals.input_utils import sparse_scipy_to_cp
from cuml.common.exceptions import NotFittedError
from cuml.internals.mixins import FMajorInputTagMixin
from cuml.internals.api_decorators import device_interop_preparation
from cuml.internals.api_decorators import enable_device_interop


IF GPUBUILD == 1:
    from enum import IntEnum
    from cython.operator cimport dereference as deref
    from cuml.decomposition.utils cimport *
    from pylibraft.common.handle cimport handle_t

    cdef extern from "cuml/decomposition/kpca.hpp" namespace "ML":
        cdef void kpcaFit(handle_t& handle,
                         float *input,
                         float *eigenvectors,
                         float *eigenvalues,
                         const paramsKPCA &prms) except +

        cdef void kpcaFit(handle_t& handle,
                         double *input,
                         double *eigenvectors,
                         double *eigenvalues,
                         const paramsKPCA &prms) except +

        cdef void kpcaTransform(handle_t& handle,
                         float *fit_input,
                         float *input,
                         float *eigenvectors,
                         float *eigenvalues,
                         float *trans_input,
                         const paramsKPCA &prms) except +
        
        cdef void kpcaTransform(handle_t& handle,
                            double *fit_input,
                            double *input,
                            double *eigenvectors,
                            double *eigenvalues,
                            double *trans_input,
                         const paramsKPCA &prms) except +

        cdef void kpcaFitTransform(handle_t& handle,
                         float *input,
                         float *eigenvectors,
                         float *eigenvalues,
                         float *trans_input,
                         const paramsKPCA &prms) except +

        cdef void kpcaFitTransform(handle_t& handle,
                         double *input,
                         double *eigenvectors,
                         double *eigenvalues,
                         double *trans_input,
                         const paramsKPCA &prms) except +

    class Solver(IntEnum):
        COV_EIG_DQ = <underlying_type_t_solver> solver.COV_EIG_DQ
        COV_EIG_JACOBI = <underlying_type_t_solver> solver.COV_EIG_JACOBI


class KernelPCA(UniversalBase,
          FMajorInputTagMixin):

    """
    =======
    TODO(TOMAS) rewrite
    =======

    
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
        >>> print(f'noise variance: {pca_float.noise_variance_}')
        noise variance: 0  0.0
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

    random_state : int / None (default = None)
        If you want results to be the same when you restart Python, select a
        state.
    eigen_solver : 'full' or 'jacobi' or 'auto' (default = 'full')
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


    For additional docs, see `scikitlearn's KernelPCA
    <http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html>`_.
    """

    _cpu_estimator_import_path = 'sklearn.decomposition.KernelPCA'
    eigenvalues_ = CumlArrayDescriptor(order='F')
    eigenvectors_ = CumlArrayDescriptor(order='F')
    trans_input_ = CumlArrayDescriptor(order='F')

    # n_components=None
    # kernel='linear' # 'linear', 'poly', 'rbf', 'sigmoid' (tanh)
    # gamma=None
    # degree=3
    # coef0=1
    # kernel_params=None # used for custom kernels
    # alpha=1.0 # not supported? For inverse
    # fit_inverse_transform=False # not supported
    # eigen_solver='auto' # full', 'jacobi' or 'auto' (default = 'full')
    # tol=0 # supported for jacobi
    # max_iter=None # supported for jacobi
    # iterated_power='auto' # not supported since it's just used for 'random' solver
    # remove_zero_eig=False # todo look into it
    # random_state=None # not supported, is only relevant for eigen_solver == ‘arpack’ or ‘randomized’.
    # copy_X=True # todo look into it. Probably support it 
    # n_jobs=None # not relevant?
    
    @device_interop_preparation
    def __init__(self, *, handle=None, n_components=None, kernel='linear', gamma=None,
                degree=3, coef0=1, kernel_params=None, alpha=1.0,
                fit_inverse_transform=False, eigen_solver='auto', tol=0,
                max_iter=None, iterated_power=15, remove_zero_eig=False,
                random_state=None, copy_X=True, verbose=False, output_type=None):
        # parameters
        super().__init__(handle=handle,
                         verbose=verbose,
                         output_type=output_type)
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.iterated_power = iterated_power
        self.n_components_ = n_components
        self.random_state = random_state
        self.eigen_solver = eigen_solver
        self.tol = tol
        self.kernel = kernel
        self.c_kernel = self._get_c_kernel(kernel)
        self.c_algorithm = self._get_algorithm_c_name(self.eigen_solver)
        self.gamma_ = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alpha = alpha
        self.fit_inverse_transform = fit_inverse_transform
        self.remove_zero_eig = remove_zero_eig

        # internal array attributes
        self.trans_input_ = None
        self.eigenvectors_ = None
        self.eigenvalues_ = None

        # This variable controls whether a sparse model was fit
        # This can be removed once there is more inter-operability
        # between cuml.array and cupy.ndarray
        self._sparse_model = None

    def _get_c_kernel(self, kernel):
        """
        Get KernelType from the kernel string.

        Parameters
        ----------
        kernel: string, ('linear', 'poly', 'rbf', or 'sigmoid')
        """
        return {
            'linear': LINEAR,
            'poly': POLYNOMIAL,
            'rbf': RBF,
            'sigmoid': TANH
        }[kernel]

    def _get_algorithm_c_name(self, algorithm):
        IF GPUBUILD == 1:
            algo_map = {
                'full': Solver.COV_EIG_DQ,
                'auto': Solver.COV_EIG_DQ,
                # 'arpack': NOT_SUPPORTED,
                # 'randomized': NOT_SUPPORTED,
                'jacobi': Solver.COV_EIG_JACOBI
            }
            if algorithm not in algo_map:
                msg = "algorithm {!r} is not supported"
                raise TypeError(msg.format(algorithm))

            return algo_map[algorithm]

    def _build_params(self, n_rows, n_cols):
        IF GPUBUILD == 1:
            cdef paramsKPCA *params = new paramsKPCA()
            params.n_components = self.n_components_ or -1
            params.n_training_samples = n_rows
            params.n_rows = n_rows
            params.n_cols = n_cols
            params.n_iterations = self.iterated_power
            params.tol = self.tol
            params.verbose = self.verbose
            params.remove_zero_eig = self.remove_zero_eig
            params.algorithm = <solver> (<underlying_type_t_solver> (
                self.c_algorithm))
            params.fit_inverse_transform = self.fit_inverse_transform
            params.kernel = self._get_kernel_params(n_cols)
            return <size_t>params

    def _initialize_arrays(self, n_components, n_rows, n_cols):

        self.eigenvalues_ = CumlArray.zeros((n_components),
                                           dtype=self.dtype)
        self.eigenvectors_ = CumlArray.zeros((n_rows, n_components),
                                           dtype=self.dtype)


    @generate_docstring(X='dense')
    @enable_device_interop
    def fit(self, X, y=None) -> "KernelPCA":
        """
        Fit the model with X. y is currently ignored.

        """
        if self.copy_X:
            self.X_fit_ = X.copy()
        self.X_m, self.n_samples_, self.n_features_in_, self.dtype = \
            input_to_cuml_array(X, check_dtype=[np.float32, np.float64])
        cdef uintptr_t _input_ptr = self.X_m.ptr
        self.feature_names_in_ = self.X_m.index

        IF GPUBUILD == 1:
            cdef paramsKPCA *params = <paramsKPCA*><size_t> \
                self._build_params(self.n_samples_, self.n_features_in_)

            # if params.n_components > self.n_features_in_:
            #     raise ValueError('Number of components should not be greater than'
            #                      'the number of columns in the data')

            # Calling _initialize_arrays, guarantees everything is CumlArray
            self._initialize_arrays(params.n_components,
                                    params.n_rows, params.n_cols)

            cdef uintptr_t eigenvectors_ptr = self.eigenvectors_.ptr

            cdef uintptr_t eigenvalues_ptr = \
                self.eigenvalues_.ptr

            cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
            if self.dtype == np.float32:
                kpcaFit(handle_[0],
                       <float*> _input_ptr,
                       <float*> eigenvectors_ptr,
                       <float*> eigenvalues_ptr,
                       deref(params))
            else:
                kpcaFit(handle_[0],
                       <double*> _input_ptr,
                       <double*> eigenvectors_ptr,
                       <double*> eigenvalues_ptr,
                       deref(params))

        # make sure the previously scheduled gpu tasks are complete before the
        # following transfers start
        self.handle.sync()

        return self

    @generate_docstring(X='dense',
                        return_values={'name': 'trans',
                                       'type': 'dense',
                                       'description': 'Transformed values',
                                       'shape': '(n_samples, n_components)'})
    @cuml.internals.api_base_return_array_skipall
    @enable_device_interop
    def fit_transform(self, X, y=None) -> CumlArray:
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set.

        """
        cuml.internals.set_api_output_type("cupy")
        if self.copy_X:
            self.X_fit_ = X.copy()
        self.X_m, self.n_samples_, self.n_features_in_, self.dtype = \
            input_to_cuml_array(X, check_dtype=[np.float32, np.float64])
        cdef uintptr_t _input_ptr = self.X_m.ptr
        self.feature_names_in_ = self.X_m.index
        IF GPUBUILD == 1:
            cdef paramsKPCA *params = <paramsKPCA*><size_t> \
                self._build_params(self.n_samples_, self.n_features_in_)

            # Calling _initialize_arrays, guarantees everything is CumlArray
            self._initialize_arrays(params.n_components,
                                    params.n_rows, params.n_cols)

            cdef uintptr_t eigenvectors_ptr = self.eigenvectors_.ptr

            cdef uintptr_t eigenvalues_ptr = \
                self.eigenvalues_.ptr


            t_input_data = \
                CumlArray.zeros((params.n_rows, params.n_components),
                                dtype=self.dtype.type, index=self.X_m.index)
            cdef uintptr_t _trans_input_ptr = t_input_data.ptr

            cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
            if self.dtype.type == np.float32:
                kpcaFitTransform(handle_[0],
                             <float*> _input_ptr,
                             <float*> eigenvectors_ptr,
                             <float*> eigenvalues_ptr,
                             <float*> _trans_input_ptr,
                             deref(params))
            else:
                kpcaFitTransform(handle_[0],
                             <double*> _input_ptr,
                             <double*> eigenvectors_ptr,
                             <double*> eigenvalues_ptr,
                             <double*> _trans_input_ptr,
                             deref(params))
            # make sure the previously scheduled gpu tasks are complete before the
            # following transfers start
            self.handle.sync()

            return t_input_data
    
    @enable_device_interop
    def transform(self, X, convert_dtype=False) -> CumlArray:
        """
        Apply dimensionality reduction to X.

        X is projected on the first principal components previously extracted
        from a training set.

        """
        self._check_is_fitted('eigenvectors_')
        cdef uintptr_t _fit_input_ptr = self.X_m.ptr

        dtype = self.eigenvectors_.dtype

        X_m, _n_rows, _n_cols, dtype = \
            input_to_cuml_array(X, check_dtype=dtype,
                                convert_to_dtype=(dtype if convert_dtype
                                                  else None),
                                check_cols=self.n_features_in_)

        cdef uintptr_t _input_ptr = X_m.ptr

        IF GPUBUILD == 1:
            cdef paramsKPCA params
            params.n_training_samples = self.n_samples_
            params.n_components = self.n_components_
            params.n_rows = _n_rows
            params.n_cols = _n_cols
            params.kernel = self._get_kernel_params(_n_cols)
            t_input_data = \
                CumlArray.zeros((params.n_rows, params.n_components),
                                dtype=dtype.type, index=X_m.index)

            cdef uintptr_t _trans_input_ptr = t_input_data.ptr
            cdef uintptr_t eigenvalues_ptr = self.eigenvalues_.ptr
            cdef uintptr_t eigenvectors_ptr = \
                self.eigenvectors_.ptr

            cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
            if dtype.type == np.float32:
                kpcaTransform(handle_[0],
                             <float*> _fit_input_ptr,
                             <float*> _input_ptr,
                             <float*> eigenvectors_ptr,
                             <float*> eigenvalues_ptr,
                             <float*> _trans_input_ptr,
                             params)
            else:
                kpcaTransform(handle_[0],
                             <double*> _fit_input_ptr,
                             <double*> _input_ptr,
                             <double*> eigenvectors_ptr,
                             <double*> eigenvalues_ptr,
                             <double*> _trans_input_ptr,
                             params)

            # make sure the previously scheduled gpu tasks are complete before the
            # following transfers start
            self.handle.sync()

            return t_input_data

    def _get_kernel_params(self, n_cols):
        """ Wrap the kernel parameters in a KernelParams object """
        cdef KernelParams _kernel_params
        if not self.gamma_:
            self.gamma_ = 1 / n_cols
        _kernel_params.kernel = self.c_kernel
        _kernel_params.degree = self.degree
        _kernel_params.gamma = self.gamma_
        _kernel_params.coef0 = self.coef0
        return _kernel_params

    def get_param_names(self):
        return super().get_param_names() + \
            ["copy_X", "iterated_power", "n_components", "eigen_solver", "tol",
                "random_state", "kernel", "gamma", "degree", "coef0", "alpha",
                "fit_inverse_transform", "remove_zero_eig", "kernel_params", "max_iter"]

    def _check_is_fitted(self, attr):
        if not hasattr(self, attr) or (getattr(self, attr) is None):
            msg = ("This instance is not fitted yet. Call 'fit' "
                   "with appropriate arguments before using this estimator.")
            raise NotFittedError(msg)


    def get_attr_names(self):
        return ['eigenvectors_', 'eigenvalues_', 'n_components_', 'X_fit_',
                'n_samples_', 'n_features_in_', 'feature_names_in_', 'gamma_']

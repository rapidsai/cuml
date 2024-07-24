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
from cuml.internals import logger

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
                         int *n_components,
                         const paramsKPCA &prms) except +

        cdef void kpcaFit(handle_t& handle,
                         double *input,
                         double *eigenvectors,
                         double *eigenvalues,
                         int *n_components,
                         const paramsKPCA &prms) except +


        cdef void kpcaFitTransform(handle_t& handle,
                         float *input,
                         float *eigenvectors,
                         float *eigenvalues,
                         float *trans_input,
                         int *n_components,
                         const paramsKPCA &prms) except +

        cdef void kpcaFitTransform(handle_t& handle,
                         double *input,
                         double *eigenvectors,
                         double *eigenvalues,
                         double *trans_input,
                         int *n_components,
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

    class Solver(IntEnum):
        COV_EIG_DQ = <underlying_type_t_solver> solver.COV_EIG_DQ
        COV_EIG_JACOBI = <underlying_type_t_solver> solver.COV_EIG_JACOBI


class KernelPCA(UniversalBase,
          FMajorInputTagMixin):

    """
    KernelPCA (Kernel Principal Component Analysis) is an extension of PCA
    that allows for non-linear dimensionality reduction through the use of
    kernel methods. It projects the data into a higher-dimensional space
    where it becomes linearly separable, and then applies PCA to capture the
    most variance in the data.

    cuML's KernelPCA expects an array-like object or cuDF DataFrame, and
    supports various kernels such as linear, polynomial, RBF, and sigmoid.

    Examples
    --------
    .. code-block:: python

        >>> # Importing KernelPCA
        >>> from cuml import KernelPCA

        >>> import cudf
        >>> import cupy as cp

        >>> gdf_float = cudf.DataFrame()
        >>> gdf_float['0'] = cp.asarray([1.0, 2.0, 5.0], dtype=cp.float32)
        >>> gdf_float['1'] = cp.asarray([4.0, 2.0, 1.0], dtype=cp.float32)
        >>> gdf_float['2'] = cp.asarray([4.0, 2.0, 1.0], dtype=cp.float32)

        >>> kpca_float = KernelPCA(n_components=2, kernel='rbf', gamma=15)
        >>> kpca_float.fit(gdf_float)
        KernelPCA()

        >>> print(f'components: {kpca_float.eigenvalues}') # doctest: +SKIP
        components: [[...], [...]]
        >>> print(f'eigen vectors: {kpca_float.eigenvectors}') # doctest: +SKIP
        eigen vectors: [...]

        >>> trans_gdf_float = kpca_float.transform(gdf_float)
        >>> print(f'Transformed: {trans_gdf_float}') # doctest: +SKIP
        Transformed: [[...], [...]]

    Parameters
    ----------
    n_components : int, optional (default=None)
        The number of components to keep. If None, all non zero eigenvalues are kept.
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'}, optional (default='linear')
        Kernel to be used in the algorithm.
    gamma : float, optional (default=None)
        Kernel coefficient for 'rbf', 'poly', and 'sigmoid'. If None, 1/n_features is used.
    degree : int, optional (default=3)
        Degree for the polynomial kernel. Ignored by other kernels.
    coef0 : float, optional (default=1)
        Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.
    kernel_params : dict, optional (default=None)
        Parameters (keyword arguments) and values for kernel passed as callable object.
    alpha : float, optional (default=1.0)
        Hyperparameter of the ridge regression that learns the inverse transform. Inverse transform not supported in cuML.
    fit_inverse_transform : bool, optional (default=False)
        Not supported in cuML.
    eigen_solver : {'auto', 'full', 'jacobi'}, optional (default='auto')
        Select eigensolver to use.
    tol : float, optional (default=0)
        Convergence tolerance for arpack.
    max_iter : int, optional (default=None)
        Not supported in available eigen solvers.
    remove_zero_eig : bool, optional (default=False)
        If True, then all components with zero eigenvalues are removed
    random_state : int or None, optional (default=None)
        Seed for the random number generator. Not supported in available eigen solvers.
    copy_X : bool, optional (default=True)
        If True, input X is copied and stored. Otherwise, X may be overwritten.
    verbose : int or bool, optional (default=False)
        Enable verbose output. If True, output is printed. If False, no output.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', 'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, optional (default=None)
        Return results and set estimator attributes to the indicated output type. If None, the output type set at the module level (`cuml.global_settings.output_type`) will be used.

    Attributes
    ----------
    eigenvectors_ : array
        Eigenvectors in the transformed space.
    eigenvalues_ : array
        Eigenvalues in the transformed space.
    X_fit_ : array
        Data used for fitting.
    gamma_ : float
        Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
    n_features_in_ : int
        Number of features in the input data.
    n_samples_ : int
        Number of samples in the input data.
    feature_names_in_ : list
        Names of the features in the input data.
    n_components_ : int
        Number of components to keep. If None, all non zero eigenvalues are kept.

    Notes
    -----
    KernelPCA (KPCA) is a non-linear extension of PCA, which allows for the capture
    of complex, non-linear structures in the data. This makes KPCA suitable for datasets
    where linear assumptions are insufficient to capture the underlying patterns.
    It employs kernel methods to project data into a higher-dimensional space where
    it becomes linearly separable, thus retaining more meaningful structure.

    **Applications of KernelPCA**

        KernelPCA is widely used for feature extraction and dimensionality reduction
        in various domains. It is particularly effective for data that exhibits
        non-linear relationships, such as in image denoising, pattern recognition,
        and pre-processing data for machine learning algorithms. It has been applied
        to gene expression data to uncover complex biological patterns, and in
        image processing to improve the performance of object recognition systems.


    For additional docs, see `scikit-learn's KernelPCA
    <http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html>`_.
    """

    _cpu_estimator_import_path = 'sklearn.decomposition.KernelPCA'
    eigenvalues_ = CumlArrayDescriptor(order='F')
    eigenvectors_ = CumlArrayDescriptor(order='F')
    trans_input_ = CumlArrayDescriptor(order='F')

    @device_interop_preparation
    def __init__(self, *, handle=None, n_components=None, kernel='linear', gamma=None,
                degree=3, coef0=1, kernel_params=None, alpha=1.0,
                fit_inverse_transform=False, eigen_solver='auto', tol=0,
                max_iter=None, iterated_power=15, remove_zero_eig=False, n_jobs=None,
                random_state=None, copy_X=True, verbose=False, output_type=None):
        if fit_inverse_transform:
            raise NotImplementedError("Inverse transform is not supported")
        if random_state is not None:
            raise NotImplementedError("Random state is not supported in available eigen solvers")
        if n_jobs is not None and n_jobs != -1:
            raise NotImplementedError("n_jobs does not apply to this algorithm")
        if max_iter is not None:
            raise NotImplementedError("max_iter is not supported in available eigen solvers. Use iterated_power for Jacobi solver")
        super().__init__(handle=handle,
                        verbose=verbose,
                        output_type=output_type)
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.iterated_power = iterated_power
        self.n_components_ = n_components
        self.remove_zero_eig = remove_zero_eig
        if remove_zero_eig:
            self.n_components = None
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

        self.trans_input_ = None
        self.eigenvectors_ = None
        self.eigenvalues_ = None

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
                'jacobi': Solver.COV_EIG_JACOBI
            }
            if algorithm not in algo_map:
                msg = "algorithm {!r} is not supported"
                raise TypeError(msg.format(algorithm))

            return algo_map[algorithm]

    def _build_params(self, n_rows, n_cols):
        IF GPUBUILD == 1:
            cdef paramsKPCA *params = new paramsKPCA()
            params.n_components = min(self.n_components_ or n_rows, n_rows)
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
        else:
            self.X_fit_ = X
        self.X_m, self.n_samples_, self.n_features_in_, self.dtype = \
            input_to_cuml_array(X, check_dtype=[np.float32, np.float64])
        if self.n_samples_ < 2: raise ValueError('n_samples must be greater than 1')
        if self.n_features_in_ < 1: raise ValueError('n_features_in_ must be greater than 0')
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
            cdef int components = <int>(self.n_components_ or -1)
            cdef int* component_ptr = &components
            cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
            if self.dtype == np.float32:
                kpcaFit(handle_[0],
                       <float*> _input_ptr,
                       <float*> eigenvectors_ptr,
                       <float*> eigenvalues_ptr,
                       component_ptr,
                       deref(params))
            else:
                kpcaFit(handle_[0],
                       <double*> _input_ptr,
                       <double*> eigenvectors_ptr,
                       <double*> eigenvalues_ptr,
                       component_ptr,
                       deref(params))
        # make sure the previously scheduled gpu tasks are complete before the
        # following transfers start
        self.handle.sync()
        self.n_components_ = components
        self.eigenvalues_ = self.eigenvalues_[:components]
        self.eigenvectors_ = self.eigenvectors_[:, :components]
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
        else:
            self.X_fit_ = X
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
            
            cdef int components = <int>(self.n_components_ or -1)
            cdef int* component_ptr = &components
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
                             component_ptr,
                             deref(params))
            else:
                kpcaFitTransform(handle_[0],
                             <double*> _input_ptr,
                             <double*> eigenvectors_ptr,
                             <double*> eigenvalues_ptr,
                             <double*> _trans_input_ptr,
                             component_ptr,
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
        if _n_cols != self.n_features_in_:
            raise ValueError("Number of columns in input must match the "
                             "number of columns in the training data")
        if _n_rows < 1:
            raise ValueError("Number of rows in input must be greater than 0")
        cdef uintptr_t _input_ptr = X_m.ptr

        IF GPUBUILD == 1:
            cdef paramsKPCA params
            params.n_training_samples = self.n_samples_
            params.n_components = len(self.eigenvalues_)
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

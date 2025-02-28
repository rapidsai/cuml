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

# distutils: language = c++

from cuml.internals.safe_imports import gpu_only_import
cudf = gpu_only_import('cudf')
from cuml.internals.safe_imports import gpu_only_import
cp = gpu_only_import('cupy')
from cuml.internals.safe_imports import cpu_only_import
np = cpu_only_import('numpy')

from cuml.internals.safe_imports import gpu_only_import_from
cuda = gpu_only_import_from('numba', 'cuda')

from cython.operator cimport dereference as deref
from libc.stdint cimport uintptr_t

import warnings
import cuml.internals
from cuml.internals.array import CumlArray
from cuml.internals.mixins import ClassifierMixin
from cuml.common.doc_utils import generate_docstring
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.internals.logger import warn
from cuml.internals.logger cimport level_enum
from pylibraft.common.handle cimport handle_t
from pylibraft.common.interruptible import cuda_interruptible
from cuml.common import input_to_cuml_array, input_to_host_array, input_to_host_array_with_sparse_support
from cuml.internals.input_utils import input_to_cupy_array, determine_array_type_full
from cuml.preprocessing import LabelEncoder
from libcpp cimport nullptr
from cuml.svm.svm_base import SVMBase
from cuml.internals.import_utils import has_sklearn
from cuml.internals.array_sparse import SparseCumlArray
from cuml.internals.api_decorators import device_interop_preparation, enable_device_interop

from sklearn.svm import SVC as skSVC
from sklearn.preprocessing import LabelBinarizer
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from cuml.internals.mem_type import MemoryType
from cuml.internals.available_devices import is_cuda_available


class cpuModelSVC(skSVC):
    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        if self.probability:
            params = self.get_params()
            params["probability"] = False

            if self.n_classes_ == 2:
                estimator = skSVC(**params)
            else:
                if self.decision_function_shape == 'ovr':
                    estimator = OneVsRestClassifier(skSVC(**params))
                elif self.decision_function_shape == 'ovo':
                    estimator = OneVsOneClassifier(skSVC(**params))
                else:
                    raise ValueError

            self.prob_svc = CalibratedClassifierCV(estimator,
                                                   cv=5,
                                                   method='sigmoid')
            self.prob_svc.fit(X, y)
        elif self.n_classes_ == 2:
            super().fit(X, y, sample_weight)
        else:
            params = self.get_params()
            if self.decision_function_shape == 'ovr':
                self.multi_class_model = OneVsRestClassifier(skSVC(**params))
            elif self.decision_function_shape == 'ovo':
                self.multi_class_model = OneVsOneClassifier(skSVC(**params))
            else:
                raise ValueError
            self.multi_class_model.fit(X, y)

    def predict(self, X):
        if self.probability:
            return self.prob_svc.predict(X)
        elif self.n_classes_ == 2:
            return super().predict(X)
        else:
            return self.multi_class_model.predict(X)

    def predict_proba(self, X):
        if self.probability:
            return self.prob_svc.predict_proba(X)
        elif self.n_classes_ == 2:
            return super().predict_proba(X)
        else:
            return self.multi_class_model.predict_proba(X)


if has_sklearn():
    from cuml.multiclass import MulticlassClassifier
    from sklearn.calibration import CalibratedClassifierCV


cdef extern from "raft/distance/distance_types.hpp" \
        namespace "raft::distance::kernels":
    enum KernelType:
        LINEAR,
        POLYNOMIAL,
        RBF,
        TANH

    cdef struct KernelParams:
        KernelType kernel
        int degree
        double gamma
        double coef0

cdef extern from "cuml/svm/svm_parameter.h" namespace "ML::SVM":
    enum SvmType:
        C_SVC,
        NU_SVC,
        EPSILON_SVR,
        NU_SVR

    cdef struct SvmParameter:
        # parameters for training
        double C
        double cache_size
        int max_iter
        int nochange_steps
        double tol
        level_enum verbosity
        double epsilon
        SvmType svmType

cdef extern from "cuml/svm/svm_model.h" namespace "ML::SVM":

    cdef cppclass SupportStorage[math_t]:
        int nnz
        int* indptr
        int* indices
        math_t* data

    cdef cppclass SvmModel[math_t]:
        # parameters of a fitted model
        int n_support
        int n_cols
        math_t b
        math_t *dual_coefs
        SupportStorage[math_t] support_matrix
        int *support_idx
        int n_classes
        math_t *unique_labels

cdef extern from "cuml/svm/svc.hpp" namespace "ML::SVM" nogil:

    cdef void svcFit[math_t](const handle_t &handle, math_t* data,
                             int n_rows, int n_cols,
                             math_t *labels,
                             const SvmParameter &param,
                             KernelParams &kernel_params,
                             SvmModel[math_t] &model,
                             const math_t *sample_weight) except +

    cdef void svcFitSparse[math_t](const handle_t &handle, int* indptr, int* indices,
                                   math_t* data, int n_rows, int n_cols, int nnz,
                                   math_t *labels,
                                   const SvmParameter &param,
                                   KernelParams &kernel_params,
                                   SvmModel[math_t] &model,
                                   const math_t *sample_weight) except +


def apply_class_weight(handle, sample_weight, class_weight, y, verbose, output_type, dtype) -> CumlArray:
    """
    Scale the sample weights with the class weights.

    Returns the modified sample weights, or None if neither class weights
    nor sample weights are defined. The returned weights are defined as

    sample_weight[i] = class_weight[y[i]] * sample_weight[i].

    Parameters:
    -----------
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model.
    sample_weight: array-like (device or host), shape = (n_samples, 1)
        sample weights or None if not given
    class_weight : dict or string (default=None)
        Weights to modify the parameter C for class i to class_weight[i]*C. The
        string 'balanced' is also accepted, in which case ``class_weight[i] =
        n_samples / (n_classes * n_samples_of_class[i])``
    y: array of floats or doubles, shape = (n_samples, 1)
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {{'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.
    dtype : dtype for sample_weights

    Returns
    --------
    sample_weight: device array shape = (n_samples, 1) or None
    """
    if class_weight is None:
        return sample_weight

    if type(y) is CumlArray:
        y_m = y
    else:
        y_m, _, _, _ = input_to_cuml_array(y, check_cols=1)

    le = LabelEncoder(handle=handle,
                      verbose=verbose,
                      output_type=output_type)
    labels = y_m.to_output(output_type='series')
    encoded_labels = cp.asarray(le.fit_transform(labels))
    n_samples = y_m.shape[0]

    # Define class weights for the encoded labels
    if class_weight == 'balanced':
        counts = cp.asnumpy(cp.bincount(encoded_labels))
        n_classes = len(counts)
        weights = n_samples / (n_classes * counts)
        class_weight = {i: weights[i] for i in range(n_classes)}
    else:
        keys = class_weight.keys()
        encoded_keys = le.transform(cudf.Series(keys)).values_host
        class_weight = {enc_key: class_weight[key]
                        for enc_key, key in zip(encoded_keys, keys)}

    if sample_weight is None:
        sample_weight = cp.ones(y_m.shape, dtype=dtype)
    else:
        sample_weight, _, _, _ = \
            input_to_cupy_array(sample_weight, convert_to_dtype=dtype,
                                check_rows=n_samples, check_cols=1)

    for label, weight in class_weight.items():
        sample_weight[encoded_labels==label] *= weight

    return sample_weight


class SVC(SVMBase,
          ClassifierMixin):
    """
    SVC (C-Support Vector Classification)

    Construct an SVC classifier for training and predictions.

    Examples
    --------
    .. code-block:: python

        >>> import cupy as cp
        >>> from cuml.svm import SVC
        >>> X = cp.array([[1,1], [2,1], [1,2], [2,2], [1,3], [2,3]],
        ...              dtype=cp.float32);
        >>> y = cp.array([-1, -1, 1, -1, 1, 1], dtype=cp.float32)
        >>> clf = SVC(kernel='poly', degree=2, gamma='auto', C=1)
        >>> clf.fit(X, y)
        SVC()
        >>> print("Predicted labels:", clf.predict(X))
        Predicted labels: [-1. -1.  1. -1.  1.  1.]

    Parameters
    ----------
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    C : float (default = 1.0)
        Penalty parameter C
    kernel : string (default='rbf')
        Specifies the kernel function. Possible options: 'linear', 'poly',
        'rbf', 'sigmoid'. Currently precomputed kernels are not supported.
    degree : int (default=3)
        Degree of polynomial kernel function.
    gamma : float or string (default = 'scale')
        Coefficient for rbf, poly, and sigmoid kernels. You can specify the
        numeric value, or use one of the following options:

        - 'auto': gamma will be set to ``1 / n_features``
        - 'scale': gamma will be se to ``1 / (n_features * X.var())``

    coef0 : float (default = 0.0)
        Independent term in kernel function, only significant for poly and
        sigmoid
    tol : float (default = 1e-3)
        Tolerance for stopping criterion.
    cache_size : float (default = 1024.0)
        Size of the kernel cache during training in MiB. Increase it to improve
        the training time, at the cost of higher memory footprint. After
        training the kernel cache is deallocated.
        During prediction, we also need a temporary space to store kernel
        matrix elements (this can be significant if n_support is large).
        The cache_size variable sets an upper limit to the prediction
        buffer as well.
    class_weight : dict or string (default=None)
        Weights to modify the parameter C for class i to class_weight[i]*C. The
        string 'balanced' is also accepted, in which case ``class_weight[i] =
        n_samples / (n_classes * n_samples_of_class[i])``
    max_iter : int (default = -1)
        Limit the number of outer iterations in the solver.
        If -1 (default) then ``max_iter=100*n_samples``
    decision_function_shape : str ('ovo' or 'ovr', default 'ovo')
        Multiclass classification strategy. ``'ovo'`` uses `OneVsOneClassifier
        <https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html>`_
        while ``'ovr'`` selects `OneVsRestClassifier
        <https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html>`_

        .. versionadded:: 25.02
          The parameter `multiclass_strategy` was renamed to `decision_function_shape`.
    nochange_steps : int (default = 1000)
        We monitor how much our stopping criteria changes during outer
        iterations. If it does not change (changes less then 1e-3*tol)
        for nochange_steps consecutive steps, then we stop training.
    output_type : {'input', 'array', 'dataframe', 'series', 'df_obj', \
        'numba', 'cupy', 'numpy', 'cudf', 'pandas'}, default=None
        Return results and set estimator attributes to the indicated output
        type. If None, the output type set at the module level
        (`cuml.global_settings.output_type`) will be used. See
        :ref:`output-data-type-configuration` for more info.
    probability: bool (default = False)
        Enable or disable probability estimates.
    random_state: int (default = None)
        Seed for random number generator (used only when probability = True).
        Currently this argument is not used and a warning will be printed if the
        user provides it.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    multiclass_strategy
        Multiclass classification strategy. ``'ovo'`` uses `OneVsOneClassifier
        <https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html>`_
        while ``'ovr'`` selects `OneVsRestClassifier
        <https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html>`_

        .. versionchanged:: 25.02
            Renamed to `decision_function_shape`. Will be removed in later versions.

    Attributes
    ----------
    n_support_ : int
        The total number of support vectors. Note: this will change in the
        future to represent number support vectors for each class (like
        in Sklearn, see https://github.com/rapidsai/cuml/issues/956 )
    support_ : int, shape = (n_support)
        Device array of support vector indices
    support_vectors_ : float, shape (n_support, n_cols)
        Device array of support vectors
    dual_coef_ : float, shape = (1, n_support)
        Device array of coefficients for support vectors
    intercept_ : float
        The constant in the decision function
    fit_status_ : int
        0 if SVM is correctly fitted
    coef_ : float, shape (1, n_cols)
        Only available for linear kernels. It is the normal of the
        hyperplane.
    classes_ : shape (`n_classes_`,)
        Array of class labels
    n_classes_ : int
        Number of classes

    Notes
    -----
    The solver uses the SMO method to fit the classifier. We use the Optimized
    Hierarchical Decomposition [1]_ variant of the SMO algorithm, similar to
    [2]_.

    For additional docs, see `scikitlearn's SVC
    <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_.

    References
    ----------
    .. [1] J. Vanek et al. A GPU-Architecture Optimized Hierarchical
       Decomposition Algorithm for Support VectorMachine Training, IEEE
       Transactions on Parallel and Distributed Systems, vol 28, no 12, 3330,
       (2017)

    .. [2] `Z. Wen et al. ThunderSVM: A Fast SVM Library on GPUs and CPUs,
       Journal of Machine Learning Research, 19, 1-5 (2018)
       <https://github.com/Xtra-Computing/thundersvm>`_

    """

    _cpu_estimator_import_path = 'cuml.svm.cpuModelSVC'

    class_weight_ = CumlArrayDescriptor(order='F')

    @device_interop_preparation
    def __init__(self, *, handle=None, C=1, kernel='rbf', degree=3,
                 gamma='scale', coef0=0.0, tol=1e-3, cache_size=1024.0,
                 max_iter=-1, nochange_steps=1000, verbose=False,
                 output_type=None, probability=False, random_state=None,
                 class_weight=None, decision_function_shape='ovo',
                 multiclass_strategy="warn"):
        super().__init__(
            handle=handle,
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            cache_size=cache_size,
            max_iter=max_iter,
            nochange_steps=nochange_steps,
            verbose=verbose,
            output_type=output_type)

        self.probability = probability
        self.random_state = random_state
        if probability and random_state is not None:
            warn("Random state is currently ignored by probabilistic SVC")
        self.class_weight = class_weight
        self.svmType = C_SVC

        self.decision_function_shape = decision_function_shape
        self.multiclass_strategy = multiclass_strategy

    @property
    @cuml.internals.api_base_return_array_skipall
    def classes_(self):
        if self.probability:
            return self.prob_svc.classes_
        elif self.n_classes_ > 2:
            return self.multiclass_svc.classes_
        else:
            return self._unique_labels_

    @classes_.setter
    def classes_(self, value):
        if self.probability:
            self.prob_svc.classes_ = value
        elif self.n_classes_ > 2:
            self.multiclass_svc.classes_ = value
        else:
            self._unique_labels_ = CumlArray.from_input(value, convert_to_dtype=self.dtype)

    @property
    @cuml.internals.api_base_return_array_skipall
    def support_(self):
        if self.n_classes_ > 2:
            estimators = self.multiclass_svc.multiclass_estimator.estimators_
            return cp.concatenate(
                [cp.asarray(cls._support_) for cls in estimators])
        else:
            return self._support_

    @support_.setter
    def support_(self, value):
        self._support_ = value

    @property
    @cuml.internals.api_base_return_array_skipall
    def intercept_(self):
        if self.n_classes_ > 2:
            estimators = self.multiclass_svc.multiclass_estimator.estimators_
            return cp.concatenate(
                [cp.asarray(cls._intercept_) for cls in estimators])
        else:
            return super()._intercept_

    @intercept_.setter
    def intercept_(self, value):
        self._intercept_ = value

    def _get_num_classes(self, y):
        """
        Determine the number of unique classes in y.
        """
        y_m, _, _, _ = input_to_cuml_array(y, check_cols=1)
        return len(cp.unique(cp.asarray(y_m)))

    def _fit_multiclass(self, X, y, sample_weight) -> "SVC":
        if sample_weight is not None:
            warn("Sample weights are currently ignored for multi class "
                 "classification")
        if not has_sklearn():
            raise RuntimeError("Scikit-learn is needed to fit multiclass SVM")

        params = self.get_params()
        strategy = params.pop('decision_function_shape', 'ovo')
        self.multiclass_svc = MulticlassClassifier(
            estimator=SVC(**params), handle=self.handle, verbose=self.verbose,
            output_type=self.output_type, strategy=strategy)
        self.multiclass_svc.fit(X, y)

        # if using one-vs-one we align support_ indices to those of
        # full dataset
        if strategy == 'ovo':
            y = cp.array(y)
            classes = cp.unique(y)
            n_classes = len(classes)
            estimator_index = 0
            # Loop through multiclass estimators and re-align support_ indices
            for i in range(n_classes):
                for j in range(i + 1, n_classes):
                    cond = cp.logical_or(y == classes[i], y == classes[j])
                    ovo_support = cp.array(
                        self.multiclass_svc.multiclass_estimator.estimators_[
                            estimator_index
                        ].support_)
                    self.multiclass_svc.multiclass_estimator.estimators_[
                        estimator_index
                    ].support_ = cp.nonzero(cond)[0][ovo_support]
                    estimator_index += 1

        self.fit_status_ = 0
        return self

    def _fit_proba(self, X, y, sample_weight) -> "SVC":
        params = self.get_params()
        params["probability"] = False

        # Ensure it always outputs numpy
        params["output_type"] = "numpy"

        # Currently CalibratedClassifierCV expects data on the host, see
        # https://github.com/rapidsai/cuml/issues/2608
        X = input_to_host_array_with_sparse_support(X)
        y = input_to_host_array(y).array

        if not has_sklearn():
            raise RuntimeError(
                "Scikit-learn is needed to use SVM probabilities")

        if self.n_classes_ == 2:
            estimator = SVC(**params)
        else:
            if self.decision_function_shape == 'ovr':
                estimator = OneVsRestClassifier(SVC(**params))
            elif self.decision_function_shape == 'ovo':
                estimator = OneVsOneClassifier(SVC(**params))
            else:
                raise ValueError

        self.prob_svc = CalibratedClassifierCV(estimator,
                                               cv=5,
                                               method='sigmoid')

        # Apply class weights to sample weights, necessary, so it doesn't crash when sample_weight is None
        sample_weight = apply_class_weight(self.handle, sample_weight, self.class_weight, y, self.verbose,
                                           self.output_type, self.dtype)

        # If sample_weight is not None, it is a cupy array, and we need to convert it to a numpy array for sklearn
        if sample_weight is not None:
            # Currently, fitting a probabilistic SVC with class weights requires at least 3 classes, otherwise the following,
            # ambiguous error is raised: ValueError: Buffer dtype mismatch, expected 'const float' but got 'double'
            if len(set(y)) < 3:
                raise ValueError("At least 3 classes are required to use probabilistic SVC with class weights.")

            # Convert cupy array to numpy array
            sample_weight = sample_weight.get()

        with cuml.internals.exit_internal_api():
            # Fit the model, sample_weight is either None or a numpy array
            self.prob_svc.fit(X, y, sample_weight=sample_weight)

        self.fit_status_ = 0
        return self

    @generate_docstring(y='dense_anydtype')
    @cuml.internals.api_base_return_any(set_output_dtype=True)
    @enable_device_interop
    def fit(self, X, y, sample_weight=None, convert_dtype=True) -> "SVC":
        """
        Fit the model with X and y.

        """
        if self.multiclass_strategy != "warn":
            self.decision_function_shape = self.multiclass_strategy
            warnings.warn('Parameter "multiclass_strategy" has been'
                          ' deprecated. Please use the'
                          ' "decision_function_shape" parameter instead.',
                          FutureWarning)

        self.n_classes_ = self._get_num_classes(y)

        # we need to check whether input X is sparse
        # In that case we don't want to make a dense copy
        _array_type, is_sparse = determine_array_type_full(X)
        self._sparse = is_sparse

        if self.probability:
            return self._fit_proba(X, y, sample_weight)

        if self.n_classes_ > 2:
            return self._fit_multiclass(X, y, sample_weight)

        if is_sparse:
            X_m = SparseCumlArray(X)
            self.n_rows = X_m.shape[0]
            self.n_features_in_ = X_m.shape[1]
            self.dtype = X_m.dtype
        else:
            X_m, self.n_rows, self.n_features_in_, self.dtype = \
                input_to_cuml_array(X, order='F')

        # Fit binary classifier
        convert_to_dtype = self.dtype if convert_dtype else None
        y_m, _, _, _ = \
            input_to_cuml_array(y, check_dtype=self.dtype,
                                convert_to_dtype=convert_to_dtype,
                                check_rows=self.n_rows, check_cols=1)

        cdef uintptr_t y_ptr = y_m.ptr

        sample_weight = apply_class_weight(self.handle, sample_weight, self.class_weight, y_m, self.verbose, self.output_type, self.dtype)
        cdef uintptr_t sample_weight_ptr = <uintptr_t> nullptr
        if sample_weight is not None:
            sample_weight_m, _, _, _ = \
                input_to_cuml_array(sample_weight, check_dtype=self.dtype,
                                    convert_to_dtype=convert_to_dtype,
                                    check_rows=self.n_rows, check_cols=1)
            sample_weight_ptr = sample_weight_m.ptr

        self._dealloc()  # delete any previously fitted model
        self.coef_ = None

        cdef KernelParams _kernel_params = self._get_kernel_params(X_m)
        cdef SvmParameter param = self._get_svm_params()
        cdef SvmModel[float] *model_f
        cdef SvmModel[double] *model_d
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        cdef int n_rows = self.n_rows
        cdef int n_cols = self.n_features_in_

        cdef int n_nnz = X_m.nnz if is_sparse else -1
        cdef uintptr_t X_indptr = X_m.indptr.ptr if is_sparse else X_m.ptr
        cdef uintptr_t X_indices = X_m.indices.ptr if is_sparse else X_m.ptr
        cdef uintptr_t X_data = X_m.data.ptr if is_sparse else X_m.ptr

        if self.dtype == np.float32:
            model_f = new SvmModel[float]()
            if is_sparse:
                with cuda_interruptible():
                    with nogil:
                        svcFitSparse(
                            deref(handle_), <int*>X_indptr, <int*>X_indices,
                            <float*>X_data, n_rows, n_cols, n_nnz,
                            <float*>y_ptr, param, _kernel_params,
                            deref(model_f), <float*>sample_weight_ptr)
            else:
                with cuda_interruptible():
                    with nogil:
                        svcFit(
                            deref(handle_), <float*>X_data, n_rows, n_cols,
                            <float*>y_ptr, param, _kernel_params,
                            deref(model_f), <float*>sample_weight_ptr)
            self._model = <uintptr_t>model_f
        elif self.dtype == np.float64:
            model_d = new SvmModel[double]()
            if is_sparse:
                with cuda_interruptible():
                    with nogil:
                        svcFitSparse(
                            deref(handle_), <int*>X_indptr, <int*>X_indices,
                            <double*>X_data, n_rows, n_cols, n_nnz,
                            <double*>y_ptr, param, _kernel_params,
                            deref(model_d), <double*>sample_weight_ptr)
            else:
                with cuda_interruptible():
                    with nogil:
                        svcFit(
                            deref(handle_), <double*>X_data, n_rows, n_cols,
                            <double*>y_ptr, param, _kernel_params,
                            deref(model_d), <double*>sample_weight_ptr)
            self._model = <uintptr_t>model_d
        else:
            raise TypeError('Input data type should be float32 or float64')

        self._unpack_model()
        self.fit_status_ = 0
        self.handle.sync()

        del X_m
        del y_m

        return self

    @generate_docstring(return_values={'name': 'preds',
                                       'type': 'dense',
                                       'description': 'Predicted values',
                                       'shape': '(n_samples, 1)'})
    @enable_device_interop
    def predict(self, X, convert_dtype=True) -> CumlArray:
        """
        Predicts the class labels for X. The returned y values are the class
        labels associated to sign(decision_function(X)).
        """

        if self.probability:
            self._check_is_fitted('prob_svc')

            X = input_to_host_array_with_sparse_support(X)

            with cuml.internals.exit_internal_api():
                preds = self.prob_svc.predict(X)
                # prob_svc has numpy output type, change it if it is necessary:
                return preds
        elif self.n_classes_ > 2:
            self._check_is_fitted('multiclass_svc')
            return self.multiclass_svc.predict(X)
        else:
            return super(SVC, self).predict(X, True, convert_dtype)

    @generate_docstring(skip_parameters_heading=True,
                        return_values={'name': 'preds',
                                       'type': 'dense',
                                       'description': 'Predicted \
                                       probabilities',
                                       'shape': '(n_samples, n_classes)'})
    @enable_device_interop
    def predict_proba(self, X, log=False) -> CumlArray:
        """
        Predicts the class probabilities for X.

        The model has to be trained with probability=True to use this method.

        Parameters
        ----------
        log: boolean (default = False)
             Whether to return log probabilities.

        """

        if self.probability:
            self._check_is_fitted('prob_svc')

            X = input_to_host_array_with_sparse_support(X)

            # Exit the internal API when calling sklearn code (forces numpy
            # conversion)
            with cuml.internals.exit_internal_api():
                preds = self.prob_svc.predict_proba(X)
                if (log):
                    preds = np.log(preds)
                # prob_svc has numpy output type, change it if it is necessary:
                return preds
        else:
            raise AttributeError("This classifier is not fitted to predict "
                                 "probabilities. Fit a new classifier with "
                                 "probability=True to enable predict_proba.")

    @generate_docstring(return_values={'name': 'preds',
                                       'type': 'dense',
                                       'description': 'Log of predicted \
                                       probabilities',
                                       'shape': '(n_samples, n_classes)'})
    @cuml.internals.api_base_return_array_skipall
    @enable_device_interop
    def predict_log_proba(self, X) -> CumlArray:
        """
        Predicts the log probabilities for X (returns log(predict_proba(x)).

        The model has to be trained with probability=True to use this method.

        """
        return self.predict_proba(X, log=True)

    @generate_docstring(return_values={'name': 'results',
                                       'type': 'dense',
                                       'description': 'Decision function \
                                       values',
                                       'shape': '(n_samples, 1)'})
    @enable_device_interop
    def decision_function(self, X) -> CumlArray:
        """
        Calculates the decision function values for X.

        """
        if self.probability:
            self._check_is_fitted('prob_svc')
            # Probabilistic SVC is an ensemble of simple SVC classifiers
            # fitted to different subset of the training data. As such, it
            # does not have a single decision function. (During prediction
            # we use the calibrated probabilities to determine the class
            # label.) Here we average the decision function value. This can
            # be useful for visualization, but predictions should be made
            # using the probabilities.
            df = np.zeros((X.shape[0],))

            with cuml.internals.exit_internal_api():
                for clf in self.prob_svc.calibrated_classifiers_:
                    df = df + clf.estimator.decision_function(X)
            df = df / len(self.prob_svc.calibrated_classifiers_)
            return df
        elif self.n_classes_ > 2:
            self._check_is_fitted('multiclass_svc')
            return self.multiclass_svc.decision_function(X)
        else:
            return super().predict(X, False)

    @classmethod
    def _get_param_names(cls):
        params = super()._get_param_names() + \
            ["probability", "random_state", "class_weight", "decision_function_shape", "multiclass_strategy"]

        # Ignore "epsilon" since its not used in the constructor
        if ("epsilon" in params):
            params.remove("epsilon")

        return params

    def get_attr_names(self):
        return super().get_attr_names() + ["classes_", "_sparse"]

    def cpu_to_gpu(self):
        self.dtype = np.float64
        self.target_dtype = np.int64
        self.probability = self._cpu_model.probability
        self.n_classes_ = self._cpu_model.n_classes_
        self.decision_function_shape = self._cpu_model.decision_function_shape

        def turn_cpu_into_gpu(cpu_est, params):
            gpu_est = SVC(**params)
            gpu_est.dtype = np.float64
            gpu_est.target_dtype = np.int64
            gpu_est.n_classes_ = 2

            # n_classes == 2 in this case
            intercept_ = -1.0 * cpu_est._intercept_
            dual_coef_ = -1.0 * cpu_est._dual_coef_

            gpu_est.n_support_ = cpu_est.n_support_.sum()

            gpu_est._intercept_ = input_to_cuml_array(
                intercept_,
                convert_to_mem_type=(MemoryType.host,
                                     MemoryType.device)[is_cuda_available()],
                convert_to_dtype=np.float64,
                order='F')[0]
            gpu_est.dual_coef_ = input_to_cuml_array(
                dual_coef_,
                convert_to_mem_type=(MemoryType.host,
                                     MemoryType.device)[is_cuda_available()],
                convert_to_dtype=np.float64,
                order='F')[0]
            gpu_est.support_ = input_to_cuml_array(
                cpu_est.support_,
                convert_to_mem_type=(MemoryType.host,
                                     MemoryType.device)[is_cuda_available()],
                convert_to_dtype=np.int32,
                order='F')[0]
            gpu_est.support_vectors_ = input_to_cuml_array(
                cpu_est.support_vectors_,
                convert_to_mem_type=(MemoryType.host,
                                     MemoryType.device)[is_cuda_available()],
                convert_to_dtype=np.float64,
                order='F')[0]
            gpu_est._unique_labels_ = input_to_cuml_array(
                np.array(cpu_est.classes_, dtype=np.float64),
                deepcopy=True,
                convert_to_mem_type=(MemoryType.host,
                                     MemoryType.device)[is_cuda_available()],
                convert_to_dtype=np.float64,
                order='F')[0]

            gpu_est._probA = cp.empty(0, dtype=np.float64)
            gpu_est._probB = cp.empty(0, dtype=np.float64)
            gpu_est._gamma = cpu_est._gamma
            gpu_est.fit_status_ = cpu_est.fit_status_
            gpu_est.n_features_in_ = cpu_est.n_features_in_
            gpu_est._sparse = cpu_est._sparse

            gpu_est._model = gpu_est._get_svm_model()
            return gpu_est

        if self.probability:
            if not hasattr(self, 'prob_svc'):
                classes = self._cpu_model.classes_

                def convert_calibrator(cpu_calibrator, params):
                    import copy
                    gpu_calibrator = copy.copy(cpu_calibrator)
                    cpu_est = cpu_calibrator.estimator
                    if isinstance(cpu_est, skSVC):
                        gpu_est = turn_cpu_into_gpu(cpu_est, params)
                    else:
                        if self.decision_function_shape == 'ovr':
                            gpu_est = OneVsRestClassifier(SVC)
                            gpu_est.label_binarizer_ = LabelBinarizer(sparse_output=True)
                            gpu_est.label_binarizer_.fit(classes)
                        elif self.decision_function_shape == 'ovo':
                            gpu_est = OneVsOneClassifier(SVC)
                            gpu_est.pairwise_indices_ = None
                        else:
                            raise ValueError
                        gpu_est.classes_ = classes
                        estimators = cpu_est.estimators_
                        gpu_est.estimators_ = [turn_cpu_into_gpu(est, params) for est in estimators]

                    gpu_calibrator.estimator = gpu_est
                    return gpu_calibrator

                params = self.get_params()
                params = {key: value for key, value, in params.items() if key in self._cpu_hyperparams}
                params["probability"] = False
                params["output_type"] = "numpy"
                self.prob_svc = CalibratedClassifierCV(SVC(**params),
                                                       cv=5,
                                                       method='sigmoid')
                self.prob_svc.classes_ = classes
                calibrators = self._cpu_model.prob_svc.calibrated_classifiers_
                self.prob_svc.calibrated_classifiers_ = [convert_calibrator(cal, params) for cal in calibrators]
        elif self.n_classes_ == 2:
            super().cpu_to_gpu()
        elif self.n_classes_ > 2:
            if not hasattr(self, 'multiclass_svc'):
                params = self.get_params()
                strategy = params.pop('decision_function_shape', 'ovo')
                self.multiclass_svc = \
                    MulticlassClassifier(estimator=SVC(**params), handle=self.handle,
                                         verbose=self.verbose, output_type=self.output_type,
                                         strategy=strategy)

                self.multiclass_svc.multiclass_estimator.classes_ = self._cpu_model.classes_
                estimators = self._cpu_model.multi_class_model.estimators_
                self.multiclass_svc.multiclass_estimator.estimators_ = [turn_cpu_into_gpu(est, params) for est in estimators]

                if strategy == 'ovr':
                    self.multiclass_svc.multiclass_estimator.label_binarizer_ = LabelBinarizer(sparse_output=True)
                    self.multiclass_svc.multiclass_estimator.label_binarizer_.fit(self._cpu_model.classes_)
                elif strategy == 'ovo':
                    self.multiclass_svc.multiclass_estimator.pairwise_indices_ = None
                else:
                    raise ValueError

    def gpu_to_cpu(self):
        self._cpu_model.n_classes_ = self.n_classes_
        self._cpu_model.decision_function_shape = self.decision_function_shape

        def turn_gpu_into_cpu(gpu_est, params):
            cpu_est = skSVC(**params)
            cpu_est.support_ = gpu_est.support_.to_output('numpy').astype(np.int32)
            cpu_est.support_vectors_ = np.ascontiguousarray(gpu_est.support_vectors_.to_output('numpy').astype(np.float64))
            cpu_est._n_support = np.array([gpu_est.n_support_, 0]).astype(np.int32)
            cpu_est._dual_coef_ = -1.0 * np.ascontiguousarray(gpu_est.dual_coef_.to_output('numpy').astype(np.float64))
            cpu_est._intercept_ = -1.0 * gpu_est.intercept_.to_output('numpy').astype(np.float64)
            cpu_est.classes_ = gpu_est.classes_.to_output('numpy').astype(np.int32)
            cpu_est.n_classes_ = 2
            cpu_est._probA = np.empty(0, dtype=np.float64)
            cpu_est._probB = np.empty(0, dtype=np.float64)
            cpu_est.fit_status_ = gpu_est.fit_status_
            cpu_est._sparse = gpu_est._sparse
            cpu_est._gamma = gpu_est._gamma
            return cpu_est

        if self.probability:
            if not hasattr(self._cpu_model, 'prob_svc'):
                def convert_calibrator(gpu_calibrator, params):
                    import copy
                    cpu_calibrator = copy.copy(gpu_calibrator)
                    gpu_est = gpu_calibrator.estimator
                    if isinstance(gpu_est, SVC):
                        cpu_est = turn_gpu_into_cpu(gpu_est, params)
                    else:
                        classes = self.classes_.to_output('numpy').astype(np.int32)
                        if self.decision_function_shape == 'ovr':
                            cpu_est = OneVsRestClassifier(skSVC)
                            cpu_est.label_binarizer_ = LabelBinarizer(sparse_output=True)
                            cpu_est.label_binarizer_.fit(classes)
                        elif self.decision_function_shape == 'ovo':
                            cpu_est = OneVsOneClassifier(skSVC)
                            cpu_est.pairwise_indices_ = None
                        else:
                            raise ValueError
                        cpu_est.classes_ = classes
                        estimators = gpu_est.estimators_
                        cpu_est.estimators_ = [turn_gpu_into_cpu(est, params) for est in estimators]

                    cpu_calibrator.estimator = cpu_est
                    return cpu_calibrator

                params = self.get_params()
                params = {key: value for key, value, in params.items() if key in self._cpu_hyperparams}
                params["probability"] = False
                self._cpu_model.prob_svc = CalibratedClassifierCV(skSVC(**params),
                                                                  cv=5,
                                                                  method='sigmoid')
                self._cpu_model.prob_svc.classes_ = self.classes_
                calibrators = self.prob_svc.calibrated_classifiers_
                self._cpu_model.prob_svc.calibrated_classifiers_ = [convert_calibrator(cal, params) for cal in calibrators]
        elif self.n_classes_ == 2:
            super().gpu_to_cpu()
        elif self.n_classes_ > 2:
            estimators = self.multiclass_svc.multiclass_estimator.estimators_
            classes = self.classes_.to_output('numpy').astype(np.int32)

            if self.decision_function_shape == 'ovr':
                self._cpu_model.multi_class_model = OneVsRestClassifier(skSVC)
                self._cpu_model.multi_class_model.label_binarizer_ = LabelBinarizer(sparse_output=True)
                self._cpu_model.multi_class_model.label_binarizer_.fit(classes)
            elif self.decision_function_shape == 'ovo':
                self._cpu_model.multi_class_model = OneVsOneClassifier(skSVC)
                self._cpu_model.multi_class_model.pairwise_indices_ = None
            else:
                raise ValueError
            self._cpu_model.multi_class_model.classes_ = classes

            params = self.get_params()
            params = {key: value for key, value, in params.items() if key in self._cpu_hyperparams}
            self._cpu_model.multi_class_model.estimators_ = [turn_gpu_into_cpu(est, params) for est in estimators]

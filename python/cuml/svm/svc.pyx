# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

import typing

import ctypes
import cudf
import cupy as cp
import numpy as np

from numba import cuda

from cython.operator cimport dereference as deref
from libc.stdint cimport uintptr_t

import cuml.internals
from cuml.common.array import CumlArray
from cuml.common.base import Base
from cuml.common.mixins import ClassifierMixin
from cuml.common.doc_utils import generate_docstring
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.logger import warn
from raft.common.handle cimport handle_t
from raft.common.interruptible import cuda_interruptible
from cuml.common import input_to_cuml_array, input_to_host_array, with_cupy_rmm
from cuml.common.input_utils import input_to_cupy_array
from cuml.preprocessing import LabelEncoder
from libcpp cimport bool, nullptr
from cuml.svm.svm_base import SVMBase
from cuml.common.import_utils import has_sklearn
from cuml.common.mixins import FMajorInputTagMixin

if has_sklearn():
    from cuml.multiclass import MulticlassClassifier
    from sklearn.calibration import CalibratedClassifierCV

cdef extern from "cuml/matrix/kernelparams.h" namespace "MLCommon::Matrix":
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
        # parameters for trainig
        double C
        double cache_size
        int max_iter
        int nochange_steps
        double tol
        int verbosity
        double epsilon
        SvmType svmType

cdef extern from "cuml/svm/svm_model.h" namespace "ML::SVM":
    cdef cppclass SvmModel[math_t]:
        # parameters of a fitted model
        int n_support
        int n_cols
        math_t b
        math_t *dual_coefs
        math_t *x_support
        int *support_idx
        int n_classes
        math_t *unique_labels

cdef extern from "cuml/svm/svc.hpp" namespace "ML::SVM" nogil:

    cdef void svcFit[math_t](const handle_t &handle, math_t *input,
                             int n_rows, int n_cols, math_t *labels,
                             const SvmParameter &param,
                             KernelParams &kernel_params,
                             SvmModel[math_t] &model,
                             const math_t *sample_weight) except+

    cdef void svcPredict[math_t](
        const handle_t &handle, math_t *input, int n_rows, int n_cols,
        KernelParams &kernel_params, const SvmModel[math_t] &model,
        math_t *preds, math_t buffer_size, bool predict_class) except +


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
        Independent term in kernel function, only signifficant for poly and
        sigmoid
    tol : float (default = 1e-3)
        Tolerance for stopping criterion.
    cache_size : float (default = 1024.0)
        Size of the kernel cache during training in MiB. Increase it to improve
        the training time, at the cost of higher memory footprint. After
        training the kernel cache is deallocated.
        During prediction, we also need a temporary space to store kernel
        matrix elements (this can be signifficant if n_support is large).
        The cache_size variable sets an upper limit to the prediction
        buffer as well.
    class_weight : dict or string (default=None)
        Weights to modify the parameter C for class i to class_weight[i]*C. The
        string 'balanced' is also accepted, in which case ``class_weight[i] =
        n_samples / (n_classes * n_samples_of_class[i])``
    max_iter : int (default = -1)
        Limit the number of outer iterations in the solver.
        If -1 (default) then ``max_iter=100*n_samples``
    multiclass_strategy : str ('ovo' or 'ovr', default 'ovo')
        Multiclass classification strategy. ``'ovo'`` uses `OneVsOneClassifier
        <https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html>`_
        while ``'ovr'`` selects `OneVsRestClassifier
        <https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html>`_
    nochange_steps : int (default = 1000)
        We monitor how much our stopping criteria changes during outer
        iterations. If it does not change (changes less then 1e-3*tol)
        for nochange_steps consecutive steps, then we stop training.
    output_type : {'input', 'cudf', 'cupy', 'numpy', 'numba'}, default=None
        Variable to control output type of the results and attributes of
        the estimator. If None, it'll inherit the output type set at the
        module level, `cuml.global_settings.output_type`.
        See :ref:`output-data-type-configuration` for more info.
    probability: bool (default = False)
        Enable or disable probability estimates.
    random_state: int (default = None)
        Seed for random number generator (used only when probability = True).
        Currently this argument is not used and a waring will be printed if the
        user provides it.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.

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
        coef_ = sum_k=1..n_support dual_coef_[k] * support_vectors[k,:]
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

    def __init__(self, *, handle=None, C=1, kernel='rbf', degree=3,
                 gamma='scale', coef0=0.0, tol=1e-3, cache_size=1024.0,
                 max_iter=-1, nochange_steps=1000, verbose=False,
                 output_type=None, probability=False, random_state=None,
                 class_weight=None, multiclass_strategy='ovo'):
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

    @cuml.internals.api_base_return_array_skipall
    def _apply_class_weight(self, sample_weight, y_m) -> CumlArray:
        """
        Scale the sample weights with the class weights.

        Returns the modified sample weights, or None if neither class weights
        nor sample weights are defined. The returned weights are defined as

        sample_weight[i] = class_weight[y[i]] * sample_weight[i].

        Parameters:
        -----------
        sample_weight: array-like (device or host), shape = (n_samples, 1)
            sample weights or None if not given
        y_m: device array of floats or doubles, shape = (n_samples, 1)
            Array of target labels already copied to the device.

        Returns
        --------
        sample_weight: device array shape = (n_samples, 1) or None
        """
        if self.class_weight is None:
            return sample_weight

        le = LabelEncoder(handle=self.handle,
                          verbose=self.verbose,
                          output_type=self.output_type)
        labels = y_m.to_output(output_type='series')
        encoded_labels = cp.asarray(le.fit_transform(labels))

        # Define class weights for the encoded labels
        if self.class_weight == 'balanced':
            counts = cp.asnumpy(cp.bincount(encoded_labels))
            n_classes = len(counts)
            n_samples = y_m.shape[0]
            weights = n_samples / (n_classes * counts)
            class_weight = {i: weights[i] for i in range(n_classes)}
        else:
            keys = self.class_weight.keys()
            keys_series = cudf.Series(keys)
            encoded_keys = le.transform(cudf.Series(keys)).values_host
            class_weight = {enc_key: self.class_weight[key]
                            for enc_key, key in zip(encoded_keys, keys)}

        if sample_weight is None:
            sample_weight = cp.ones(y_m.shape, dtype=self.dtype)
        else:
            sample_weight, _, _, _ = \
                input_to_cupy_array(sample_weight, convert_to_dtype=self.dtype,
                                    check_rows=self.n_rows, check_cols=1)

        for label, weight in class_weight.items():
            sample_weight[encoded_labels==label] *= weight

        return sample_weight

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
        strategy = params.pop('multiclass_strategy', 'ovo')

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

        self._fit_status_ = 0
        return self

    def _fit_proba(self, X, y, samle_weight) -> "SVC":
        params = self.get_params()
        params["probability"] = False

        # Ensure it always outputs numpy
        params["output_type"] = "numpy"

        # Currently CalibratedClassifierCV expects data on the host, see
        # https://github.com/rapidsai/cuml/issues/2608
        X, _, _, _, _ = input_to_host_array(X)
        y, _, _, _, _ = input_to_host_array(y)

        if not has_sklearn():
            raise RuntimeError(
                "Scikit-learn is needed to use SVM probabilities")

        self.prob_svc = CalibratedClassifierCV(SVC(**params),
                                               cv=5,
                                               method='sigmoid')

        with cuml.internals.exit_internal_api():
            self.prob_svc.fit(X, y)
        self._fit_status_ = 0
        return self

    @generate_docstring(y='dense_anydtype')
    @cuml.internals.api_base_return_any(set_output_dtype=True)
    def fit(self, X, y, sample_weight=None, convert_dtype=True) -> "SVC":
        """
        Fit the model with X and y.

        """

        self.n_classes_ = self._get_num_classes(y)

        if self.probability:
            return self._fit_proba(X, y, sample_weight)

        if self.n_classes_ > 2:
            return self._fit_multiclass(X, y, sample_weight)

        # Fit binary classifier
        X_m, self.n_rows, self.n_cols, self.dtype = \
            input_to_cuml_array(X, order='F')

        cdef uintptr_t X_ptr = X_m.ptr
        convert_to_dtype = self.dtype if convert_dtype else None
        y_m, _, _, _ = \
            input_to_cuml_array(y, check_dtype=self.dtype,
                                convert_to_dtype=convert_to_dtype,
                                check_rows=self.n_rows, check_cols=1)

        cdef uintptr_t y_ptr = y_m.ptr

        sample_weight = self._apply_class_weight(sample_weight, y_m)
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
        cdef int n_cols = self.n_cols
        if self.dtype == np.float32:
            model_f = new SvmModel[float]()
            with cuda_interruptible():
                with nogil:
                    svcFit(
                        deref(handle_), <float*>X_ptr, n_rows,
                        n_cols, <float*>y_ptr, param, _kernel_params,
                        deref(model_f), <float*>sample_weight_ptr)
            self._model = <uintptr_t>model_f
        elif self.dtype == np.float64:
            model_d = new SvmModel[double]()
            with cuda_interruptible():
                with nogil:
                    svcFit(
                        deref(handle_), <double*>X_ptr, n_rows,
                        n_cols, <double*>y_ptr, param, _kernel_params,
                        deref(model_d), <double*>sample_weight_ptr)
            self._model = <uintptr_t>model_d
        else:
            raise TypeError('Input data type should be float32 or float64')

        self._unpack_model()
        self._fit_status_ = 0
        self.handle.sync()

        del X_m
        del y_m

        return self

    @generate_docstring(return_values={'name': 'preds',
                                       'type': 'dense',
                                       'description': 'Predicted values',
                                       'shape': '(n_samples, 1)'})
    def predict(self, X, convert_dtype=True) -> CumlArray:
        """
        Predicts the class labels for X. The returned y values are the class
        labels associated to sign(decision_function(X)).
        """

        if self.probability:
            self._check_is_fitted('prob_svc')

            X, _, _, _, _ = input_to_host_array(X)

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

            X, _, _, _, _ = input_to_host_array(X)

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
                    df = df + clf.base_estimator.decision_function(X)
            df = df / len(self.prob_svc.calibrated_classifiers_)
            return df
        elif self.n_classes_ > 2:
            self._check_is_fitted('multiclass_svc')
            return self.multiclass_svc.decision_function(X)
        else:
            return super().predict(X, False)

    def get_param_names(self):
        params = super().get_param_names() + \
            ["probability", "random_state", "class_weight",
             "multiclass_strategy"]

        # Ignore "epsilon" since its not used in the constructor
        if ("epsilon" in params):
            params.remove("epsilon")

        return params

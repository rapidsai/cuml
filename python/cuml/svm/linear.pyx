# Copyright (c) 2021, NVIDIA CORPORATION.
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

import re
import typing
import numpy as np
import cupy as cp
import cuml

import cuml.internals
from cython.operator cimport dereference as deref
from cuml.common.array import CumlArray
from cuml.common.base import Base
from cuml.common.mixins import ClassifierMixin, RegressorMixin
from cuml.common.doc_utils import generate_docstring
from cuml.raft.common.handle cimport handle_t, _Stream
from rmm._lib.device_uvector cimport device_uvector
from cuml.common import input_to_cuml_array, input_to_host_array
from libc.stdint cimport uintptr_t
from libcpp cimport bool as cppbool, nullptr

from cuml.common.import_utils import has_sklearn
if has_sklearn():
    from cuml.multiclass import MulticlassClassifier
    from sklearn.calibration import CalibratedClassifierCV

__all__ = ['LinearSVC', 'LinearSVR']

cdef extern from "cuml/svm/linear.hpp" namespace "ML::SVM":

    cdef enum Penalty "ML::SVM::LinearSVMParams::Penalty":
        L1 "ML::SVM::LinearSVMParams::L1"
        L2 "ML::SVM::LinearSVMParams::L2"

    cdef enum Loss "ML::SVM::LinearSVMParams::Loss":
        HINGE "ML::SVM::LinearSVMParams::\
            HINGE"
        SQUARED_HINGE "ML::SVM::LinearSVMParams::\
            SQUARED_HINGE"
        EPSILON_INSENSITIVE "ML::SVM::LinearSVMParams::\
            EPSILON_INSENSITIVE"
        SQUARED_EPSILON_INSENSITIVE "ML::SVM::LinearSVMParams::\
            SQUARED_EPSILON_INSENSITIVE"

    cdef struct LinearSVMParams:

        Penalty penalty
        Loss loss
        cppbool fit_intercept
        cppbool penalized_intercept
        cppbool probability
        int max_iter
        int linesearch_max_iter
        int lbfgs_memory
        int verbose
        double C
        double grad_tol
        double change_tol
        double svr_sensitivity
        double H1_value

    cdef cppclass LinearSVMModel[T]:
        const handle_t& handle
        const int nRows
        const int nCols
        device_uvector[T] classes
        device_uvector[T] probScale
        device_uvector[T] w

        LinearSVMModel(
            const handle_t& handle,
            const LinearSVMParams params,
            const T* X,
            const int nRows,
            const int nCols,
            const T* y,
            const T* sampleWeight) except +
        void predict(
            const T* X, const int nRows, const int nCols, T* out) except +
        void predict_proba(
            const T* X, const int nRows, const int nCols,
            const cppbool log, T* out) except +


cdef union LinearSVMModelPtr:
    uintptr_t untyped
    LinearSVMModel[float] * float32
    LinearSVMModel[double] * float64


cdef class LSVMPWrapper_:
    cdef readonly dict params

    def __cinit__(self, **kwargs):
        cdef LinearSVMParams ps
        self.params = ps

    def _getparam(self, key):
        return self.params[key]

    def _setparam(self, key, val):
        self.params[key] = val

    def __init__(self, **kwargs):
        allowed_keys = set(self.get_param_names())
        for key, val in kwargs.items():
            if key in allowed_keys:
                setattr(self, key, val)

    def get_param_names(self):
        cdef LinearSVMParams ps
        return ps.keys()


# Here we can do custom conversion for selected properties.
class LSVMPWrapper(LSVMPWrapper_):

    @property
    def penalty(self) -> str:
        if self._getparam('penalty') == Penalty.L1:
            return "l1"
        if self._getparam('penalty') == Penalty.L2:
            return "l2"
        raise ValueError(
            f"Unknown penalty enum value: {self._getparam('penalty')}")

    @penalty.setter
    def penalty(self, penalty: str):
        if penalty == "l1":
            self._setparam('penalty', Penalty.L1)
        elif penalty == "l2":
            self._setparam('penalty', Penalty.L2)
        else:
            raise ValueError(f"Unknown penalty string value: {penalty}")

    @property
    def loss(self) -> str:
        loss = self._getparam('loss')
        if loss == Loss.HINGE:
            return "hinge"
        if loss == Loss.SQUARED_HINGE:
            return "squared_hinge"
        if loss == Loss.EPSILON_INSENSITIVE:
            return "epsilon_insensitive"
        if loss == Loss.SQUARED_EPSILON_INSENSITIVE:
            return "squared_epsilon_insensitive"
        raise ValueError(f"Unknown loss enum value: {loss}")

    @loss.setter
    def loss(self, loss: str):
        if loss == "hinge":
            self._setparam('loss', Loss.HINGE)
        elif loss == "squared_hinge":
            self._setparam('loss', Loss.SQUARED_HINGE)
        elif loss == "epsilon_insensitive":
            self._setparam('loss', Loss.EPSILON_INSENSITIVE)
        elif loss == "squared_epsilon_insensitive":
            self._setparam('loss', Loss.SQUARED_EPSILON_INSENSITIVE)
        else:
            raise ValueError(f"Unknown loss string value: {loss}")


# Add properties for parameters with a trivial conversion
def __add_prop(prop_name):
    setattr(LSVMPWrapper, prop_name, property(
        lambda self: self._getparam(prop_name),
        lambda self, value: self._setparam(prop_name, value)
    ))


for prop_name in LSVMPWrapper().get_param_names():
    if not hasattr(LSVMPWrapper, prop_name):
        __add_prop(prop_name)
del __add_prop

LinearSVM_defaults = LSVMPWrapper()
'''Default parameter values for LinearSVM, re-exported from C++.'''

cdef class LinearSVM:
    cdef public object handle
    cdef readonly object dtype
    cdef LinearSVMModelPtr model

    cdef void reset_model(self):
        if self.model.untyped != 0:
            if self.dtype == np.float32:
                del self.model.float32
            elif self.dtype == np.float64:
                del self.model.float64
        self.model.untyped = 0
        self.dtype = None

    def __cinit__(
            self,
            handle: typing.Optional[cuml.Handle] = None,
            tol: typing.Optional[float] = None,
            **kwargs):
        self.handle = handle if handle is not None else cuml.Handle()
        self.reset_model()
        # special treatment of argument 'tol'
        if tol is not None:
            default_to_ratio = \
                LinearSVM_defaults.change_tol / LinearSVM_defaults.grad_tol
            self.grad_tol = tol
            self.change_tol = tol * default_to_ratio

    def __dealloc__(self):
        self.reset_model()

    def __getnewargs_ex__(self):
        return (), {'handle': self.handle}

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    def __init__(self, **kwargs):
        # ignore special arguments (they are set during __cinit__ anyway)
        kwargs.pop('tol', None)
        # All arguments are optional (they have defaults),
        # yet we need to check for unused arguments
        allowed_keys = set(self.get_param_names())
        super_keys = getattr(super(), 'get_param_names', lambda: [])()
        remaining_kwargs = {}
        for key, val in kwargs.items():
            if key not in allowed_keys or key in super_keys:
                remaining_kwargs[key] = val
                continue
            if val is None:
                continue
            allowed_keys.remove(key)
            setattr(self, key, val)

        # set defaults
        for key in allowed_keys:
            setattr(self, key, getattr(LinearSVM_defaults, key, None))

        super().__init__(**remaining_kwargs)

    @property
    def n_classes_(self):
        if self.dtype == np.float32:
            return self.model.float32.classes.size()
        if self.dtype == np.float64:
            return self.model.float32.classes.size()
        raise AttributeError('Train the model first')

    @property
    def intercept_(self):
        cl = self.n_classes_
        if cl == 2:
            cl = 1
        if not self.fit_intercept:
            return CumlArray.zeros(shape=(cl, ), dtype=self.dtype)
        cdef uintptr_t b_ptr
        k = 0
        if self.dtype == np.float32:
            k = self.model.float32.nCols
            b_ptr = <uintptr_t> (self.model.float32.w.data() + <int>(k*cl))
        elif self.dtype == np.float64:
            k = self.model.float64.nCols
            b_ptr = <uintptr_t> (self.model.float64.w.data() + <int>(k*cl))
        else:
            raise AttributeError('Train the model first')
        return CumlArray(
            b_ptr, dtype=self.dtype, shape=(cl, ), owner=self, order='C'
            ).to_output(output_type='numpy')

    @property
    def coef_(self):
        cdef uintptr_t w_ptr
        k = 0
        cl = self.n_classes_
        if cl == 2:
            cl = 1
        if self.dtype == np.float32:
            k = self.model.float32.nCols
            w_ptr = <uintptr_t> self.model.float32.w.data()
        elif self.dtype == np.float64:
            k = self.model.float64.nCols
            w_ptr = <uintptr_t> self.model.float64.w.data()
        else:
            raise AttributeError('Train the model first')
        return CumlArray(  # NB: on the c-side it's shape is C-major (k, cl)
            w_ptr, dtype=self.dtype, shape=(cl, k), owner=self, order='F'
            ).to_output(output_type='numpy')

    @property
    def classes_(self):
        cdef uintptr_t c_ptr
        cl = self.n_classes_
        if cl == 2:
            cl = 1
        if self.dtype == np.float32:
            c_ptr = <uintptr_t> self.model.float32.classes.data()
        elif self.dtype == np.float64:
            c_ptr = <uintptr_t> self.model.float64.classes.data()
        else:
            raise AttributeError('Train the model first')
        return CumlArray(
            c_ptr, dtype=self.dtype, shape=(cl, ), owner=self, order='F'
            ).to_output(output_type='numpy')

    def get_param_names(self):
        return LinearSVM_defaults.get_param_names()

    def fit(self, X, y, sample_weight=None, convert_dtype=True):
        """
        Fit the model with X and y.

        """
        self.reset_model()

        # Fit binary classifier
        X_m, n_rows, n_cols, self.dtype = \
            input_to_cuml_array(X, order='F')

        cdef uintptr_t X_ptr = X_m.ptr
        convert_to_dtype = self.dtype if convert_dtype else None
        y_m, _, _, _ = \
            input_to_cuml_array(y, check_dtype=self.dtype,
                                convert_to_dtype=convert_to_dtype,
                                check_rows=n_rows, check_cols=1)

        cdef uintptr_t y_ptr = y_m.ptr

        # sample_weight = self._apply_class_weight(sample_weight, y_m)
        cdef uintptr_t sample_weight_ptr = <uintptr_t> nullptr
        if sample_weight is not None:
            sample_weight_m, _, _, _ = \
                input_to_cuml_array(sample_weight, check_dtype=self.dtype,
                                    convert_to_dtype=convert_to_dtype,
                                    check_rows=n_rows, check_cols=1)
            sample_weight_ptr = sample_weight_m.ptr

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        cdef LinearSVMParams params = LSVMPWrapper(
            **{k: getattr(self, k) for k in self.get_param_names()}
        ).params
        if self.dtype == np.float32:
            self.model.float32 = new LinearSVMModel[float](
                deref(handle_),
                params,
                <const float*>X_ptr,
                n_rows, n_cols,
                <const float*>y_ptr,
                <const float*>sample_weight_ptr)
        elif self.dtype == np.float64:
            self.model.float64 = new LinearSVMModel[double](
                deref(handle_),
                params,
                <const double*>X_ptr,
                n_rows, n_cols,
                <const double*>y_ptr,
                <const double*>sample_weight_ptr)
        else:
            raise TypeError('Input data type should be float32 or float64')

        del X_m
        del y_m

    def predict(self, X, convert_dtype=True) -> CumlArray:
        convert_to_dtype = self.dtype if convert_dtype else None
        X_m, n_rows, n_cols, _ = \
            input_to_cuml_array(X, check_dtype=self.dtype,
                                convert_to_dtype=convert_to_dtype)

        cdef uintptr_t X_ptr = X_m.ptr

        y = CumlArray.empty(shape=(n_rows,), dtype=X_m.dtype, order='F')
        cdef uintptr_t y_ptr = y.ptr

        if self.dtype == np.float32:
            self.model.float32.predict(
                <const float*>X_ptr, n_rows, n_cols,
                <float*>y_ptr)
        elif self.dtype == np.float64:
            self.model.float64.predict(
                <const double*>X_ptr, n_rows, n_cols,
                <double*>y_ptr)
        else:
            raise TypeError('Input data type should be float32 or float64')

        return y

    # def predict_proba(self, X, log=False, convert_dtype=True) -> CumlArray:
    #     convert_to_dtype = self.dtype if convert_dtype else None
    #     X_m, n_rows, n_cols, _ = \
    #         input_to_cuml_array(X, check_dtype=self.dtype,
    #                             convert_to_dtype=convert_to_dtype)

    #     cdef uintptr_t X_ptr = X_m.ptr

    #     y = CumlArray.empty(shape=(n_rows, 2), dtype=X_m.dtype, order='F')
    #     cdef uintptr_t y_ptr = y.ptr

    #     if self.dtype == np.float32:
    #         self.model.float32.predict_proba(
    #             <const float*>X_ptr, n_rows, n_cols, log,
    #             (<float*>y_ptr) + <int>n_rows)
    #     elif self.dtype == np.float64:
    #         self.model.float64.predict_proba(
    #             <const double*>X_ptr, n_rows, n_cols, log,
    #             (<double*>y_ptr) + <int>n_rows)
    #     else:
    #         raise TypeError('Input data type should be float32 or float64')

    #     y[:, 0] = CumlArray.ones(shape=(n_rows, ), dtype=y.dtype) - y[:, 1]
    #     return y


class LinearSVC(LinearSVM, Base, ClassifierMixin):
    '''
    LinearSVC (Support Vector Classification with the linear kernel)

    Construct a linear SVM classifier for training and predictions.

    Examples
    --------
    .. code-block:: python

        import numpy as np
        from cuml.svm import LinearSVC
        X = np.array([[1,1], [2,1], [1,2], [2,2], [1,3], [2,3]],
                        dtype=np.float32);
        y = np.array([0, 0, 1, 0, 1, 1], dtype=np.float32)
        clf = LinearSVC(loss='squared_hinge', penalty='l1', C=1)
        clf.fit(X, y)
        print("Predicted labels:", clf.predict(X))

    Output:

    .. code-block:: none

        Predicted labels: [0. 0. 1. 0. 1. 1.]

    Parameters
    ----------
    handle : cuml.Handle
        Specifies the cuml.handle that holds internal CUDA state for
        computations in this model. Most importantly, this specifies the CUDA
        stream that will be used for the model's computations, so users can
        run different models concurrently in different streams by creating
        handles in several streams.
        If it is None, a new one is created.
    penalty : {{'l1', 'l2'}} (default = '{LinearSVM_defaults.penalty}')
        The regularization term of the target function.
    loss : {LinearSVC.REGISTERED_LOSSES} (default = 'squared_hinge')
        The loss term of the target function.
    fit_intercept : {LinearSVM_defaults.fit_intercept.__class__.__name__ \
            } (default = {LinearSVM_defaults.fit_intercept})
        Whether to fit the bias term. Set to False if you expect that the
        data is already centered.
    penalized_intercept : { \
            LinearSVM_defaults.penalized_intercept.__class__.__name__ \
            } (default = {LinearSVM_defaults.penalized_intercept})
        When true, the bias term is treated the same way as other features;
        i.e. it's penalized by the regularization term of the target function.
        Enabling this feature forces an extra copying the input data X.
    max_iter : {LinearSVM_defaults.max_iter.__class__.__name__ \
            } (default = {LinearSVM_defaults.max_iter})
        Maximum number of iterations for the underlying solver.
    linesearch_max_iter : { \
            LinearSVM_defaults.linesearch_max_iter.__class__.__name__ \
            } (default = {LinearSVM_defaults.linesearch_max_iter})
        Maximum number of linesearch (inner loop) iterations for
        the underlying (QN) solver.
    lbfgs_memory : { \
            LinearSVM_defaults.lbfgs_memory.__class__.__name__ \
            } (default = {LinearSVM_defaults.lbfgs_memory})
        Number of vectors approximating the hessian for the underlying QN
        solver (l-bfgs).
    verbose : { \
            LinearSVM_defaults.verbose.__class__.__name__ \
            } (default = {LinearSVM_defaults.verbose})
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    C : {LinearSVM_defaults.C.__class__.__name__ \
            } (default = {LinearSVM_defaults.C})
        The constant scaling factor of the loss term in the target formula
          `F(X, y) = penalty(X) + C * loss(X, y)`.
    grad_tol : {LinearSVM_defaults.grad_tol.__class__.__name__ \
            } (default = {LinearSVM_defaults.grad_tol})
        The threshold on the gradient for the underlying QN solver.
    change_tol : {LinearSVM_defaults.change_tol.__class__.__name__ \
            } (default = {LinearSVM_defaults.change_tol})
        The threshold on the function change for the underlying QN solver.
    tol : Optional[float] (default = None)
        Tolerance for the stopping criterion.
        This is a helper transient parameter that, when present, sets both
        `grad_tol` and `change_tol` to the same value. When any of the two
        `***_tol` parameters are passed as well, they take the precedence.
    H1_value : {LinearSVM_defaults.H1_value.__class__.__name__ \
            } (default = {LinearSVM_defaults.H1_value})
        he value considered 'one' in the binary classification problem.
        This value is converted into `1.0` during training, whereas all the
        other values in the training target data (`y`)
        are converted into `-1.0`.
    probability: bool (default = False)
        Enable or disable probability estimates.
    multiclass_strategy : {{'ovo' or 'ovr'}} (default = 'ovo')
        Multiclass classification strategy. ``'ovo'`` uses `OneVsOneClassifier
        <https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsOneClassifier.html>`_
        while ``'ovr'`` selects `OneVsRestClassifier
        <https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html>`_
    output_type : {{'input', 'cudf', 'cupy', 'numpy', 'numba'}} (default=None)
        Variable to control output type of the results and attributes of
        the estimator. If None, it'll inherit the output type set at the
        module level, `cuml.global_settings.output_type`.
        See :ref:`output-data-type-configuration` for more info.

    Attributes
    ----------
    intercept_ : float
        The constant in the decision function
    coef_ : float, shape (1, n_cols)
        Only available for linear kernels. It is the normal of the
        hyperplane.
        coef_ = sum_k=1..n_support dual_coef_[k] * support_vectors[k,:]
    classes_: shape (n_classes_,)
        Array of class labels
    n_classes_ : int
        Number of classes

    Notes
    -----
    The model uses the quasi-newton (QN) solver to find the solution in the
    primal space. Thus, in contrast to generic :class:`SVC<cuml.svm.SVC>`
    model, it does not compute the support coefficients/vectors.

    Check the solver's documentation for more details
    :class:`Quasi-Newton (L-BFGS/OWL-QN)<cuml.QN>`.

    For additional docs, see `scikitlearn's SVC
    <https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html>`_.
    '''

    REGISTERED_LOSSES = set([
        'hinge',
        'squared_hinge'])

    def __init__(self, *args, **kwargs):
        # set classification-specific defaults
        if 'loss' not in kwargs:
            kwargs['loss'] = 'squared_hinge'
        if 'multiclass_strategy' not in kwargs:
            kwargs['multiclass_strategy'] = 'ovr'
        if 'probability' not in kwargs:
            kwargs['probability'] = False

        super().__init__(*args, **kwargs)


    @property
    def loss(self):
        return self.__loss

    @loss.setter
    def loss(self, loss: str):
        if loss not in self.REGISTERED_LOSSES:
            raise ValueError(
                f"Classification loss type "
                f"must be one of {self.REGISTERED_LOSSES}, "
                f"but given '{loss}'.")
        self.__loss = loss

    def fit(self, X, y, sample_weight=None, convert_dtype=True):
        self._set_output_type(X)
        return LinearSVM.fit(
                self,
                X, y,
                sample_weight=sample_weight,
                convert_dtype=convert_dtype)
        # X_m, n_rows, n_cols, dtype = \
        #     input_to_cuml_array(X, order='F')

        # cdef uintptr_t X_ptr = X_m.ptr
        # convert_to_dtype = dtype if convert_dtype else None
        # y_m, _, _, _ = \
        #     input_to_cuml_array(y, check_dtype=dtype,
        #                         convert_to_dtype=convert_to_dtype,
        #                         check_rows=n_rows, check_cols=1)

        # self._classes_ = cp.unique(cp.asarray(y_m))
        # if self._classes_.shape[0] == 1:
        #     raise ValueError(
        #         "Only one unique target value is found, training is not possible.")

        # if self._classes_.shape[0] == 2:
        #     self.H1_value = self._classes_[1].item()
        #     LinearSVM.fit(
        #             self,
        #             X_m, y_m,
        #             sample_weight=sample_weight,
        #             convert_dtype=convert_dtype)
        #     del X_m, y_m
        #     return self

        # # multiclass
        # params = self.get_params()
        # params["probability"] = True
        # self.multiclass_svc = MulticlassClassifier(
        #     estimator=LinearSVC(**params),
        #     handle=self.handle, verbose=self.verbose,
        #     output_type=self.output_type,
        #     strategy=self.multiclass_strategy)
        # self.multiclass_svc.fit(X_m, y_m)
        # return self.multiclass_svc

    def predict(self, X, convert_dtype=True) -> CumlArray:
        out_type = self._get_output_type(X)
        return LinearSVM.predict(
                self,
                X,
                convert_dtype=convert_dtype).to_output(out_type)
        # if self._classes_.shape[0] > 2:
        #     res = self.multiclass_svc.predict(X)
        # else:
        #     res = LinearSVM.predict(self, X, convert_dtype)
        # return res.to_output(out_type)


    # def predict_proba(self, X, log=False, convert_dtype=True) -> CumlArray:
    #     """
    #     Predicts the class probabilities for X.

    #     The model has to be trained with probability=True to use this method.

    #     Parameters
    #     ----------
    #     log: boolean (default = False)
    #             Whether to return log probabilities.

    #     """
    #     out_type = self._get_output_type(X)
    #     if self._classes_.shape[0] > 2:
    #         res = self.multiclass_svc.predict_proba(X, log)
    #     else:
    #         res = LinearSVM.predict_proba(self, X, log, convert_dtype)
    #     return res.to_output(out_type)

    def get_param_names(self):
        return [
            "handle",
            "verbose",
            "output_type",
            'penalty',
            'loss',
            'fit_intercept',
            'penalized_intercept',
            'probability',
            'max_iter',
            'linesearch_max_iter',
            'lbfgs_memory',
            'C',
            'grad_tol',
            'change_tol',
            'H1_value',
            'multiclass_strategy',
        ]


# Format docstring to see the re-exported defaults etc.
LinearSVC.__doc__ = \
    re.sub(r"\{ *([^ ]+) *\}", r"{\1}", LinearSVC.__doc__).format(**locals())


class LinearSVR(LinearSVM, Base, RegressorMixin):
    '''LinearSVR'''

    REGISTERED_LOSSES = set([
        'epsilon_insensitive',
        'squared_epsilon_insensitive'])

    def __init__(self, *args, **kwargs):
        # set regression-specific defaults
        if 'loss' not in kwargs:
            kwargs['loss'] = 'epsilon_insensitive'

        super().__init__(*args, **kwargs)

    @property
    def loss(self):
        return self.__loss

    @loss.setter
    def loss(self, loss: str):
        if loss not in self.REGISTERED_LOSSES:
            raise ValueError(
                f"Regression loss type "
                f"must be one of {self.REGISTERED_LOSSES}, "
                f"but given '{loss}'.")
        self.__loss = loss

    def get_param_names(self):
        return [
            "handle",
            "verbose",
            "output_type",
            'penalty',
            'loss',
            'fit_intercept',
            'penalized_intercept',
            'max_iter',
            'linesearch_max_iter',
            'lbfgs_memory',
            'C',
            'grad_tol',
            'change_tol',
            'svr_sensitivity'
        ]


# Format docstring to see the re-exported defaults etc.
LinearSVR.__doc__ = \
    re.sub(r"\{ *([^ ]+) *\}", r"{\1}", LinearSVR.__doc__).format(**locals())


# [WIP]
# !! Checkout requirements in python/cuml/common/base.pyx
#     I must satisfy them to integrate into the cuml ML algo infrastructure
#
# NB: [FEA] scikit-learn based meta estimators
#     https://github.com/rapidsai/cuml/issues/2876
#
#     [REVIEW] Sklearn-based preprocessin
#     https://github.com/rapidsai/cuml/pull/2645
#
#     [FEA] Implement probability calibration using device arrays
#     https://github.com/rapidsai/cuml/issues/2608
#
# Need:
#   1. classes a.k.a. unique_labels
#   2. multiclass strategy
#   3. decide where transform labels into multiclass 1, -1
#   4. Also food for thought:
#
# https://scikit-learn.org/stable/modules/svm.html#svm-multi-class
# https://github.com/scikit-learn/scikit-learn/blob/ccf8e749bac06aa456d8de2d06dbe0d8c507ac3f/sklearn/multiclass.py#L598

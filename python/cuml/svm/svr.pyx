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

import ctypes
import cudf
import cupy
import numpy as np

from numba import cuda

from cython.operator cimport dereference as deref
from libc.stdint cimport uintptr_t

from cuml.common.array import CumlArray
from cuml.common.base import Base
from cuml.common.mixins import RegressorMixin
from cuml.common.doc_utils import generate_docstring
from cuml.metrics import r2_score
from raft.common.handle cimport handle_t
from cuml.common import input_to_cuml_array
from libcpp cimport bool, nullptr
from cuml.svm.svm_base import SVMBase

cdef extern from "cuml/matrix/kernelparams.h" namespace "MLCommon::Matrix":
    enum KernelType:
        LINEAR, POLYNOMIAL, RBF, TANH

    cdef struct KernelParams:
        KernelType kernel
        int degree
        double gamma
        double coef0

cdef extern from "cuml/svm/svm_parameter.h" namespace "ML::SVM":
    enum SvmType:
        C_SVC, NU_SVC, EPSILON_SVR, NU_SVR

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

cdef extern from "cuml/svm/svc.hpp" namespace "ML::SVM":

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

    cdef void svmFreeBuffers[math_t](const handle_t &handle,
                                     SvmModel[math_t] &m) except +

cdef extern from "cuml/svm/svr.hpp" namespace "ML::SVM" nogil:

    cdef void svrFit[math_t](const handle_t &handle, math_t *X,
                             int n_rows, int n_cols, math_t *y,
                             const SvmParameter &param,
                             KernelParams &kernel_params,
                             SvmModel[math_t] &model,
                             const math_t *sample_weight) except+


class SVR(SVMBase, RegressorMixin):
    """
    SVR (Epsilon Support Vector Regression)

    Construct an SVC classifier for training and predictions.

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
    epsilon: float (default = 0.1)
        epsilon parameter of the epsiron-SVR model. There is no penalty
        associated to points that are predicted within the epsilon-tube
        around the target values.
    cache_size : float (default = 1024.0)
        Size of the kernel cache during training in MiB. Increase it to improve
        the training time, at the cost of higher memory footprint. After
        training the kernel cache is deallocated.
        During prediction, we also need a temporary space to store kernel
        matrix elements (this can be signifficant if n_support is large).
        The cache_size variable sets an upper limit to the prediction
        buffer as well.
    max_iter : int (default = -1)
        Limit the number of outer iterations in the solver.
        If -1 (default) then ``max_iter=100*n_samples``
    nochange_steps : int (default = 1000)
        We monitor how much our stopping criteria changes during outer
        iterations. If it does not change (changes less then 1e-3*tol)
        for nochange_steps consecutive steps, then we stop training.
    verbose : int or boolean, default=False
        Sets logging level. It must be one of `cuml.common.logger.level_*`.
        See :ref:`verbosity-levels` for more info.
    output_type : {'input', 'cudf', 'cupy', 'numpy', 'numba'}, default=None
        Variable to control output type of the results and attributes of
        the estimator. If None, it'll inherit the output type set at the
        module level, `cuml.global_settings.output_type`.
        See :ref:`output-data-type-configuration` for more info.

    Attributes
    ----------
    n_support_ : int
        The total number of support vectors. Note: this will change in the
        future to represent number support vectors for each class (like
        in Sklearn, see Issue #956)
    support_ : int, shape = [n_support]
        Device array of suppurt vector indices
    support_vectors_ : float, shape [n_support, n_cols]
        Device array of support vectors
    dual_coef_ : float, shape = [1, n_support]
        Device array of coefficients for support vectors
    intercept_ : int
        The constant in the decision function
    fit_status_ : int
        0 if SVM is correctly fitted
    coef_ : float, shape [1, n_cols]
        Only available for linear kernels. It is the normal of the
        hyperplane.
        ``coef_ = sum_k=1..n_support dual_coef_[k] * support_vectors[k,:]``

    Notes
    -----

    For additional docs, see `Scikit-learn's SVR
    <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html>`_.

    The solver uses the SMO method to fit the regressor. We use the Optimized
    Hierarchical Decomposition [1]_ variant of the SMO algorithm, similar to
    [2]_

    References
    ----------

    .. [1] J. Vanek et al. A GPU-Architecture Optimized Hierarchical
           Decomposition Algorithm for Support VectorMachine Training, IEEE
           Transactions on Parallel and Distributed Systems, vol 28, no 12,
           3330, (2017)

    .. [2] `Z. Wen et al. ThunderSVM: A Fast SVM Library on GPUs and CPUs,
           Journal of Machine Learning Research, 19, 1-5 (2018)
           <https://github.com/Xtra-Computing/thundersvm>`_

    Examples
    --------
    .. code-block:: python


        >>> import cupy as cp
        >>> from cuml.svm import SVR
        >>> X = cp.array([[1], [2], [3], [4], [5]], dtype=cp.float32)
        >>> y = cp.array([1.1, 4, 5, 3.9, 1.], dtype = cp.float32)
        >>> reg = SVR(kernel='rbf', gamma='scale', C=10, epsilon=0.1)
        >>> reg.fit(X, y)
        SVR()
        >>> print("Predicted values:", reg.predict(X)) # doctest: +SKIP
        Predicted values: [1.200474 3.8999617 5.100488 3.7995374 1.0995375]

    """
    def __init__(self, *, handle=None, C=1, kernel='rbf', degree=3,
                 gamma='scale', coef0=0.0, tol=1e-3, epsilon=0.1,
                 cache_size=1024.0, max_iter=-1, nochange_steps=1000,
                 verbose=False, output_type=None):
        super().__init__(
            handle=handle,
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            tol=tol,
            epsilon=epsilon,
            cache_size=cache_size,
            max_iter=max_iter,
            nochange_steps=nochange_steps,
            verbose=verbose,
            output_type=output_type,
        )

        self.svmType = EPSILON_SVR

    @generate_docstring()
    def fit(self, X, y, sample_weight=None, convert_dtype=True) -> "SVR":
        """
        Fit the model with X and y.

        """
        cdef uintptr_t X_ptr, y_ptr

        X_m, self.n_rows, self.n_cols, self.dtype = \
            input_to_cuml_array(X, order='F')
        X_ptr = X_m.ptr

        convert_to_dtype = self.dtype if convert_dtype else None
        y_m, _, _, _ = \
            input_to_cuml_array(y, check_dtype=self.dtype,
                                convert_to_dtype=convert_to_dtype,
                                check_rows=self.n_rows, check_cols=1)

        y_ptr = y_m.ptr

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

        if self.dtype == np.float32:
            model_f = new SvmModel[float]()
            svrFit(handle_[0], <float*>X_ptr, <int>self.n_rows,
                   <int>self.n_cols, <float*>y_ptr, param, _kernel_params,
                   model_f[0], <float*>sample_weight_ptr)
            self._model = <uintptr_t>model_f
        elif self.dtype == np.float64:
            model_d = new SvmModel[double]()
            svrFit(handle_[0], <double*>X_ptr, <int>self.n_rows,
                   <int>self.n_cols, <double*>y_ptr, param, _kernel_params,
                   model_d[0], <double*>sample_weight_ptr)
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
        Predicts the values for X.

        """

        return super(SVR, self).predict(X, False, convert_dtype)

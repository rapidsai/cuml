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

import cupy as cp
import numpy as np

from libc.stdint cimport uintptr_t

import cuml.internals
from cuml.common import input_to_cuml_array
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.doc_utils import generate_docstring
from cuml.common.sparse_utils import is_sparse
from cuml.internals.array import CumlArray
from cuml.internals.array_sparse import SparseCumlArray
from cuml.internals.base import Base
from cuml.internals.global_settings import GlobalSettings
from cuml.internals.mixins import FMajorInputTagMixin

from libcpp cimport bool

from cuml.metrics import accuracy_score

from pylibraft.common.handle cimport handle_t


cdef extern from "cuml/linear_model/glm.hpp" namespace "ML::GLM" nogil:

    cdef enum qn_loss_type "ML::GLM::qn_loss_type":
        QN_LOSS_LOGISTIC "ML::GLM::QN_LOSS_LOGISTIC"
        QN_LOSS_SQUARED  "ML::GLM::QN_LOSS_SQUARED"
        QN_LOSS_SOFTMAX  "ML::GLM::QN_LOSS_SOFTMAX"
        QN_LOSS_SVC_L1   "ML::GLM::QN_LOSS_SVC_L1"
        QN_LOSS_SVC_L2   "ML::GLM::QN_LOSS_SVC_L2"
        QN_LOSS_SVR_L1   "ML::GLM::QN_LOSS_SVR_L1"
        QN_LOSS_SVR_L2   "ML::GLM::QN_LOSS_SVR_L2"
        QN_LOSS_ABS      "ML::GLM::QN_LOSS_ABS"
        QN_LOSS_UNKNOWN  "ML::GLM::QN_LOSS_UNKNOWN"

    cdef struct qn_params:
        qn_loss_type loss
        double penalty_l1
        double penalty_l2
        double grad_tol
        double change_tol
        int max_iter
        int linesearch_max_iter
        int lbfgs_memory
        int verbose
        bool fit_intercept
        bool penalty_normalized

    void qnFit[T, I](
        const handle_t& cuml_handle,
        const qn_params& pams,
        T *X,
        bool X_col_major,
        T *y,
        I N,
        I D,
        I C,
        T *w0,
        T *f,
        int *num_iters,
        T *sample_weight) except +

    void qnFitSparse[T, I](
        const handle_t& cuml_handle,
        const qn_params& pams,
        T *X_values,
        I *X_cols,
        I *X_row_ids,
        I X_nnz,
        T *y,
        I N,
        I D,
        I C,
        T *w0,
        T *f,
        int *num_iters,
        T *sample_weight) except +

    void qnDecisionFunction[T, I](
        const handle_t& cuml_handle,
        const qn_params& pams,
        T *X,
        bool X_col_major,
        I N,
        I D,
        I C,
        T *params,
        T *scores) except +

    void qnDecisionFunctionSparse[T, I](
        const handle_t& cuml_handle,
        const qn_params& pams,
        T *X_values,
        I *X_cols,
        I *X_row_ids,
        I X_nnz,
        I N,
        I D,
        I C,
        T *params,
        T *scores) except +

    void qnPredict[T, I](
        const handle_t& cuml_handle,
        const qn_params& pams,
        T *X,
        bool X_col_major,
        I N,
        I D,
        I C,
        T *params,
        T *preds) except +

    void qnPredictSparse[T, I](
        const handle_t& cuml_handle,
        const qn_params& pams,
        T *X_values,
        I *X_cols,
        I *X_row_ids,
        I X_nnz,
        I N,
        I D,
        I C,
        T *params,
        T *preds) except +


class StructWrapper(type):
    '''Define a property for each key in `get_param_defaults`,
        for which there is no explicit property defined in the class.
    '''
    def __new__(cls, name, bases, attrs):
        def add_prop(prop_name):
            setattr(x, prop_name, property(
                lambda self: self._getparam(prop_name),
                lambda self, value: self._setparam(prop_name, value)
            ))

        x = super().__new__(cls, name, bases, attrs)

        for prop_name in getattr(x, 'get_param_defaults', lambda: {})():
            if not hasattr(x, prop_name):
                add_prop(prop_name)
        del add_prop

        return x


class StructParams(metaclass=StructWrapper):
    params: dict

    def __new__(cls, *args, **kwargs):
        x = object.__new__(cls)
        x.params = cls.get_param_defaults().copy()
        return x

    def __init__(self, **kwargs):
        allowed_keys = set(self._get_param_names())
        for key, val in kwargs.items():
            if key in allowed_keys:
                setattr(self, key, val)

    def _getparam(self, key):
        return self.params[key]

    def _setparam(self, key, val):
        self.params[key] = val

    @classmethod
    def _get_param_names(cls):
        return cls.get_param_defaults().keys()

    def __str__(self):
        return type(self).__name__ + str(self.params)


class QNParams(StructParams):

    @staticmethod
    def get_param_defaults():
        cdef qn_params ps
        return ps

    @property
    def loss(self) -> str:
        loss = self._getparam('loss')
        if loss == qn_loss_type.QN_LOSS_LOGISTIC:
            return "sigmoid"
        if loss == qn_loss_type.QN_LOSS_SQUARED:
            return "l2"
        if loss == qn_loss_type.QN_LOSS_SOFTMAX:
            return "softmax"
        if loss == qn_loss_type.QN_LOSS_SVC_L1:
            return "svc_l1"
        if loss == qn_loss_type.QN_LOSS_SVC_L2:
            return "svc_l2"
        if loss == qn_loss_type.QN_LOSS_SVR_L1:
            return "svr_l1"
        if loss == qn_loss_type.QN_LOSS_SVR_L2:
            return "svr_l2"
        if loss == qn_loss_type.QN_LOSS_ABS:
            return "l1"
        else:
            raise ValueError(f"Unknown loss enum value: {loss}")

    @loss.setter
    def loss(self, loss: str):
        if loss in {"sigmoid", "logistic"}:
            self._setparam('loss', qn_loss_type.QN_LOSS_LOGISTIC)
        elif loss == "softmax":
            self._setparam('loss', qn_loss_type.QN_LOSS_SOFTMAX)
        elif loss in {"normal", "l2"}:
            self._setparam('loss', qn_loss_type.QN_LOSS_SQUARED)
        elif loss == "l1":
            self._setparam('loss', qn_loss_type.QN_LOSS_ABS)
        elif loss == "svc_l1":
            self._setparam('loss', qn_loss_type.QN_LOSS_SVC_L1)
        elif loss == "svc_l2":
            self._setparam('loss', qn_loss_type.QN_LOSS_SVC_L2)
        elif loss == "svr_l1":
            self._setparam('loss', qn_loss_type.QN_LOSS_SVR_L1)
        elif loss == "svr_l2":
            self._setparam('loss', qn_loss_type.QN_LOSS_SVR_L2)
        else:
            raise ValueError(f"Unknown loss string value: {loss}")


class QN(Base,
         FMajorInputTagMixin):
    """
    Quasi-Newton methods are used to either find zeroes or local maxima
    and minima of functions, and used by this class to optimize a cost
    function.

    Two algorithms are implemented underneath cuML's QN class, and which one
    is executed depends on the following rule:

      * Orthant-Wise Limited Memory Quasi-Newton (OWL-QN) if there is l1
        regularization

      * Limited Memory BFGS (L-BFGS) otherwise.

    cuML's QN class can take array-like objects, either in host as
    NumPy arrays or in device (as Numba or __cuda_array_interface__ compliant).

    Examples
    --------
    .. code-block:: python

        >>> import cudf
        >>> import cupy as cp

        >>> # Both import methods supported
        >>> # from cuml import QN
        >>> from cuml.solvers import QN

        >>> X = cudf.DataFrame()
        >>> X['col1'] = cp.array([1,1,2,2], dtype=cp.float32)
        >>> X['col2'] = cp.array([1,2,2,3], dtype=cp.float32)
        >>> y = cudf.Series(cp.array([0.0, 0.0, 1.0, 1.0], dtype=cp.float32) )

        >>> solver = QN()
        >>> solver.fit(X,y)
        QN()

        >>> # Note: for now, the coefficients also include the intercept in the
        >>> # last position if fit_intercept=True
        >>> print(solver.coef_) # doctest: +SKIP
        0   37.371...
        1   0.949...
        dtype: float32
        >>> print(solver.intercept_) # doctest: +SKIP
        0   -57.738...
        >>> X_new = cudf.DataFrame()
        >>> X_new['col1'] = cp.array([1,5], dtype=cp.float32)
        >>> X_new['col2'] = cp.array([2,5], dtype=cp.float32)
        >>> preds = solver.predict(X_new)
        >>> print(preds)
        0    0.0
        1    1.0
        dtype: float32

    Parameters
    ----------
    loss: 'sigmoid', 'softmax', 'l1', 'l2', 'svc_l1', 'svc_l2', 'svr_l1', \
        'svr_l2' (default = 'sigmoid').
        'sigmoid' loss used for single class logistic regression;
        'softmax' loss used for multiclass logistic regression;
        'l1'/'l2' loss used for regression.
    fit_intercept: boolean (default = True)
        If True, the model tries to correct for the global mean of y.
        If False, the model expects that you have centered the data.
    l1_strength: float (default = 0.0)
        l1 regularization strength (if non-zero, will run OWL-QN, else L-BFGS).
        Use `penalty_normalized` to control whether the solver divides this
        by the sample size.
    l2_strength: float (default = 0.0)
        l2 regularization strength.
        Use `penalty_normalized` to control whether the solver divides this
        by the sample size.
    max_iter: int (default = 1000)
        Maximum number of iterations taken for the solvers to converge.
    tol: float (default = 1e-4)
        The training process will stop if

        `norm(current_loss_grad) <= tol * max(current_loss, tol)`.

        This differs slightly from the `gtol`-controlled stopping condition in
        `scipy.optimize.minimize(method='L-BFGS-B')
        <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html>`_:

        `norm(current_loss_projected_grad) <= gtol`.

        Note, `sklearn.LogisticRegression()
        <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_
        uses the sum of softmax/logistic loss over the input data, whereas cuML
        uses the average. As a result, Scikit-learn's loss is usually
        `sample_size` times larger than cuML's.
        To account for the differences you may divide the `tol` by the sample
        size; this would ensure that the cuML solver does not stop earlier than
        the Scikit-learn solver.
    delta: Optional[float] (default = None)
        The training process will stop if

        `abs(current_loss - previous_loss) <= delta * max(current_loss, tol)`.

        When `None`, it's set to `tol * 0.01`; when `0`, the check is disabled.
        Given the current step `k`, parameter `previous_loss` here is the loss
        at the step `k - p`, where `p` is a small positive integer set
        internally.

        Note, this parameter corresponds to `ftol` in
        `scipy.optimize.minimize(method='L-BFGS-B')
        <https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html>`_,
        which is set by default to a minuscule `2.2e-9` and is not exposed in
        `sklearn.LogisticRegression()
        <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_.
        This condition is meant to protect the solver against doing vanishingly
        small linesearch steps or zigzagging.
        You may choose to set `delta = 0` to make sure the cuML solver does
        not stop earlier than the Scikit-learn solver.
    linesearch_max_iter: int (default = 50)
        Max number of linesearch iterations per outer iteration of the
        algorithm.
    lbfgs_memory: int (default = 5)
        Rank of the lbfgs inverse-Hessian approximation. Method will use
        O(lbfgs_memory * D) memory.
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
    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit as
        initialization, otherwise, just erase the previous solution.
    penalty_normalized : bool, default=True
        When set to True, l1 and l2 parameters are divided by the sample size.
        This flag can be used to achieve a behavior compatible with other
        implementations, such as sklearn's.

    Attributes
    ----------
    coef_ : array, shape (n_classes, n_features)
        The estimated coefficients for the linear regression model.
        Note: shape is (n_classes, n_features + 1) if fit_intercept = True.
    intercept_ : array (n_classes, 1)
        The independent term. If `fit_intercept` is False, will be 0.

    Notes
    -----
       This class contains implementations of two popular Quasi-Newton methods:

         - Limited-memory Broyden Fletcher Goldfarb Shanno (L-BFGS) [Nocedal,
           Wright - Numerical Optimization (1999)]

         - `Orthant-wise limited-memory quasi-newton (OWL-QN)
           [Andrew, Gao - ICML 2007]
           <https://www.microsoft.com/en-us/research/publication/scalable-training-of-l1-regularized-log-linear-models/>`_
    """

    _coef_ = CumlArrayDescriptor()
    intercept_ = CumlArrayDescriptor()

    def __init__(self, *, loss='sigmoid', fit_intercept=True,
                 l1_strength=0.0, l2_strength=0.0, max_iter=1000, tol=1e-4,
                 delta=None, linesearch_max_iter=50, lbfgs_memory=5,
                 verbose=False, handle=None, output_type=None,
                 warm_start=False, penalty_normalized=True):

        super().__init__(handle=handle,
                         verbose=verbose,
                         output_type=output_type)

        self.fit_intercept = fit_intercept
        self.l1_strength = l1_strength
        self.l2_strength = l2_strength
        self.max_iter = max_iter
        self.tol = tol
        self.delta = delta
        self.linesearch_max_iter = linesearch_max_iter
        self.lbfgs_memory = lbfgs_memory
        self.num_iter = 0
        self._coef_ = None
        self.intercept_ = None
        self.warm_start = warm_start
        self.penalty_normalized = penalty_normalized
        self.loss = loss

    @property
    @cuml.internals.api_base_return_array_skipall
    def coef_(self):
        if self._coef_ is None:
            return None
        if self.fit_intercept:
            val = self._coef_[0:-1]
        else:
            val = self._coef_
        val = val.to_output('array')
        val = val.T
        return val

    @coef_.setter
    def coef_(self, value):
        value = value.to_output('array').T
        if self.fit_intercept:
            value = GlobalSettings().xpy.vstack([value, self.intercept_])
        value, _, _, _ = input_to_cuml_array(value)
        self._coef_ = value

    @generate_docstring(X='dense_sparse')
    def fit(self, X, y, sample_weight=None, convert_dtype=True) -> "QN":
        """
        Fit the model with X and y.

        """
        sparse_input = is_sparse(X)
        # Handle sparse inputs
        if sparse_input:
            X_m = SparseCumlArray(X, convert_index=np.int32)
            n_rows, self.n_cols = X_m.shape
            self.dtype = X_m.dtype

        # Handle dense inputs
        else:
            X_m, n_rows, self.n_cols, self.dtype = input_to_cuml_array(
                X,
                convert_to_dtype=(np.float32 if convert_dtype
                                  else None),
                check_dtype=[np.float32, np.float64],
                order='K'
            )

        y_m, _, _, _ = input_to_cuml_array(
            y, check_dtype=self.dtype,
            convert_to_dtype=(self.dtype if convert_dtype else None),
            check_rows=n_rows, check_cols=1
        )
        cdef uintptr_t _y_ptr = y_m.ptr

        cdef uintptr_t _sample_weight_ptr = 0
        if sample_weight is not None:
            sample_weight, _, _, _ = \
                input_to_cuml_array(sample_weight,
                                    check_dtype=self.dtype,
                                    check_rows=n_rows, check_cols=1,
                                    convert_to_dtype=(self.dtype
                                                      if convert_dtype
                                                      else None))
            _sample_weight_ptr = sample_weight.ptr

        self.qnparams = QNParams(
            loss=self.loss,
            penalty_l1=self.l1_strength,
            penalty_l2=self.l2_strength,
            grad_tol=self.tol,
            change_tol=self.delta
            if self.delta is not None else (self.tol * 0.01),
            max_iter=self.max_iter,
            linesearch_max_iter=self.linesearch_max_iter,
            lbfgs_memory=self.lbfgs_memory,
            verbose=self.verbose,
            fit_intercept=self.fit_intercept,
            penalty_normalized=self.penalty_normalized
        )

        cdef qn_params qnpams = self.qnparams.params

        solves_classification = qnpams.loss in {
            qn_loss_type.QN_LOSS_LOGISTIC,
            qn_loss_type.QN_LOSS_SOFTMAX,
            qn_loss_type.QN_LOSS_SVC_L1,
            qn_loss_type.QN_LOSS_SVC_L2
        }
        solves_multiclass = qnpams.loss in {
            qn_loss_type.QN_LOSS_SOFTMAX
        }

        if solves_classification:
            self._num_classes = len(cp.unique(y_m))
        else:
            self._num_classes = 1

        if not solves_multiclass and self._num_classes > 2:
            raise ValueError(
                f"The selected solver ({self.loss}) does not support"
                f" more than 2 classes ({self._num_classes} discovered).")

        if qnpams.loss == qn_loss_type.QN_LOSS_SOFTMAX and self._num_classes <= 2:
            raise ValueError(
                "Two classes or less cannot be trained with softmax (multinomial)."
            )

        if solves_classification and not solves_multiclass:
            self._num_classes_dim = self._num_classes - 1
        else:
            self._num_classes_dim = self._num_classes

        if self.fit_intercept:
            coef_size = (self.n_cols + 1, self._num_classes_dim)
        else:
            coef_size = (self.n_cols, self._num_classes_dim)

        if self._coef_ is None or not self.warm_start:
            self._coef_ = CumlArray.zeros(
                coef_size, dtype=self.dtype, order='C')

        cdef uintptr_t _coef_ptr = self._coef_.ptr

        cdef float objective32
        cdef double objective64
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()

        cdef int num_iters

        if self.dtype == np.float32:
            if sparse_input:
                qnFitSparse[float, int](
                    handle_[0],
                    qnpams,
                    <float*><uintptr_t> X_m.data.ptr,
                    <int*><uintptr_t> X_m.indices.ptr,
                    <int*><uintptr_t> X_m.indptr.ptr,
                    <int> X_m.nnz,
                    <float*> _y_ptr,
                    <int> n_rows,
                    <int> self.n_cols,
                    <int> self._num_classes,
                    <float*> _coef_ptr,
                    <float*> &objective32,
                    <int*> &num_iters,
                    <float*> _sample_weight_ptr)

            else:
                qnFit[float, int](
                    handle_[0],
                    qnpams,
                    <float*><uintptr_t> X_m.ptr,
                    <bool> _is_col_major(X_m),
                    <float*> _y_ptr,
                    <int> n_rows,
                    <int> self.n_cols,
                    <int> self._num_classes,
                    <float*> _coef_ptr,
                    <float*> &objective32,
                    <int*> &num_iters,
                    <float*> _sample_weight_ptr)

            self.objective = objective32

        else:
            if sparse_input:
                qnFitSparse[double, int](
                    handle_[0],
                    qnpams,
                    <double*><uintptr_t> X_m.data.ptr,
                    <int*><uintptr_t> X_m.indices.ptr,
                    <int*><uintptr_t> X_m.indptr.ptr,
                    <int> X_m.nnz,
                    <double*> _y_ptr,
                    <int> n_rows,
                    <int> self.n_cols,
                    <int> self._num_classes,
                    <double*> _coef_ptr,
                    <double*> &objective64,
                    <int*> &num_iters,
                    <double*> _sample_weight_ptr)

            else:
                qnFit[double, int](
                    handle_[0],
                    qnpams,
                    <double*><uintptr_t> X_m.ptr,
                    <bool> _is_col_major(X_m),
                    <double*> _y_ptr,
                    <int> n_rows,
                    <int> self.n_cols,
                    <int> self._num_classes,
                    <double*> _coef_ptr,
                    <double*> &objective64,
                    <int*> &num_iters,
                    <double*> _sample_weight_ptr)

            self.objective = objective64

        self.num_iters = num_iters

        self._calc_intercept()

        self.handle.sync()

        del X_m
        del y_m

        return self

    @cuml.internals.api_base_return_array_skipall
    def _decision_function(self, X, convert_dtype=True) -> CumlArray:
        """
        Gives confidence score for X

        Parameters
        ----------
        X : array-like (device or host) shape = (n_samples, n_features)
            Dense matrix (floats or doubles) of shape (n_samples, n_features).
            Acceptable formats: cuDF DataFrame, NumPy ndarray, Numba device
            ndarray, cuda array interface compliant array like CuPy

        convert_dtype : bool, optional (default = False)
            When set to True, the predict method will, when necessary, convert
            the input to the data type which was used to train the model. This
            will increase memory used for the method.

        Returns
        ----------
        y: array-like (device)
            Dense matrix (floats or doubles) of shape (n_samples,), or
            (n_samples, n_classes) if more than 2 classes.

        """
        coefs = self.coef_
        dtype = coefs.dtype
        _num_classes_dim, n_cols = coefs.shape

        sparse_input = is_sparse(X)
        # Handle sparse inputs
        if sparse_input:
            X_m = SparseCumlArray(
                X,
                convert_to_dtype=(dtype if convert_dtype else None),
                convert_index=np.int32
            )
            n_rows, n_cols = X_m.shape
            dtype = X_m.dtype

        # Handle dense inputs
        else:
            X_m, n_rows, n_cols, dtype = input_to_cuml_array(
                X, check_dtype=dtype,
                convert_to_dtype=(dtype if convert_dtype else None),
                check_cols=n_cols,
                order='K'
            )

        if _num_classes_dim > 1:
            shape = (_num_classes_dim, n_rows)
        else:
            shape = (n_rows,)
        scores = CumlArray.zeros(shape=shape, dtype=dtype, order='F')

        cdef uintptr_t _coef_ptr = self._coef_.ptr
        cdef uintptr_t _scores_ptr = scores.ptr

        if not hasattr(self, 'qnparams'):
            self.qnparams = QNParams(
                loss=self.loss,
                penalty_l1=self.l1_strength,
                penalty_l2=self.l2_strength,
                grad_tol=self.tol,
                change_tol=self.delta
                if self.delta is not None else (self.tol * 0.01),
                max_iter=self.max_iter,
                linesearch_max_iter=self.linesearch_max_iter,
                lbfgs_memory=self.lbfgs_memory,
                verbose=self.verbose,
                fit_intercept=self.fit_intercept,
                penalty_normalized=self.penalty_normalized
            )

        _num_classes = self.get_num_classes(_num_classes_dim)

        cdef qn_params qnpams = self.qnparams.params
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        if dtype == np.float32:
            if sparse_input:
                qnDecisionFunctionSparse[float, int](
                    handle_[0],
                    qnpams,
                    <float*><uintptr_t> X_m.data.ptr,
                    <int*><uintptr_t> X_m.indices.ptr,
                    <int*><uintptr_t> X_m.indptr.ptr,
                    <int> X_m.nnz,
                    <int> n_rows,
                    <int> n_cols,
                    <int> _num_classes,
                    <float*> _coef_ptr,
                    <float*> _scores_ptr)
            else:
                qnDecisionFunction[float, int](
                    handle_[0],
                    qnpams,
                    <float*><uintptr_t> X_m.ptr,
                    <bool> _is_col_major(X_m),
                    <int> n_rows,
                    <int> n_cols,
                    <int> _num_classes,
                    <float*> _coef_ptr,
                    <float*> _scores_ptr)

        else:
            if sparse_input:
                qnDecisionFunctionSparse[double, int](
                    handle_[0],
                    qnpams,
                    <double*><uintptr_t> X_m.data.ptr,
                    <int*><uintptr_t> X_m.indices.ptr,
                    <int*><uintptr_t> X_m.indptr.ptr,
                    <int> X_m.nnz,
                    <int> n_rows,
                    <int> n_cols,
                    <int> _num_classes,
                    <double*> _coef_ptr,
                    <double*> _scores_ptr)
            else:
                qnDecisionFunction[double, int](
                    handle_[0],
                    qnpams,
                    <double*><uintptr_t> X_m.ptr,
                    <bool> _is_col_major(X_m),
                    <int> n_rows,
                    <int> n_cols,
                    <int> _num_classes,
                    <double*> _coef_ptr,
                    <double*> _scores_ptr)

        self._calc_intercept()

        self.handle.sync()

        del X_m

        return scores.to_output("array").T

    @generate_docstring(
        X='dense_sparse',
        return_values={
            'name': 'preds',
            'type': 'dense',
            'description': 'Predicted values',
            'shape': '(n_samples, 1)'
        })
    @cuml.internals.api_base_return_array(get_output_dtype=True)
    def predict(self, X, convert_dtype=True) -> CumlArray:
        """
        Predicts the y for X.

        """
        coefs = self.coef_
        dtype = coefs.dtype
        _num_classes_dim, n_cols = coefs.shape

        sparse_input = is_sparse(X)

        # Handle sparse inputs
        if sparse_input:
            X_m = SparseCumlArray(
                X,
                convert_to_dtype=(dtype if convert_dtype else None),
                convert_index=np.int32
            )
            n_rows, n_cols = X_m.shape

        # Handle dense inputs
        else:
            X_m, n_rows, n_cols, dtype = input_to_cuml_array(
                X, check_dtype=dtype,
                convert_to_dtype=(dtype if convert_dtype else None),
                check_cols=n_cols,
                order='K'
            )

        preds = CumlArray.zeros(shape=n_rows, dtype=dtype,
                                index=X_m.index)
        cdef uintptr_t _coef_ptr = self._coef_.ptr
        cdef uintptr_t _pred_ptr = preds.ptr

        # temporary fix for dask-sql empty partitions
        if(n_rows == 0):
            return preds

        if not hasattr(self, 'qnparams'):
            self.qnparams = QNParams(
                loss=self.loss,
                penalty_l1=self.l1_strength,
                penalty_l2=self.l2_strength,
                grad_tol=self.tol,
                change_tol=self.delta
                if self.delta is not None else (self.tol * 0.01),
                max_iter=self.max_iter,
                linesearch_max_iter=self.linesearch_max_iter,
                lbfgs_memory=self.lbfgs_memory,
                verbose=self.verbose,
                fit_intercept=self.fit_intercept,
                penalty_normalized=self.penalty_normalized
            )

        _num_classes = self.get_num_classes(_num_classes_dim)
        cdef qn_params qnpams = self.qnparams.params
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        if dtype == np.float32:
            if sparse_input:
                qnPredictSparse[float, int](
                    handle_[0],
                    qnpams,
                    <float*><uintptr_t> X_m.data.ptr,
                    <int*><uintptr_t> X_m.indices.ptr,
                    <int*><uintptr_t> X_m.indptr.ptr,
                    <int> X_m.nnz,
                    <int> n_rows,
                    <int> n_cols,
                    <int> _num_classes,
                    <float*> _coef_ptr,
                    <float*> _pred_ptr)
            else:
                qnPredict[float, int](
                    handle_[0],
                    qnpams,
                    <float*><uintptr_t> X_m.ptr,
                    <bool> _is_col_major(X_m),
                    <int> n_rows,
                    <int> n_cols,
                    <int> _num_classes,
                    <float*> _coef_ptr,
                    <float*> _pred_ptr)

        else:
            if sparse_input:
                qnPredictSparse[double, int](
                    handle_[0],
                    qnpams,
                    <double*><uintptr_t> X_m.data.ptr,
                    <int*><uintptr_t> X_m.indices.ptr,
                    <int*><uintptr_t> X_m.indptr.ptr,
                    <int> X_m.nnz,
                    <int> n_rows,
                    <int> n_cols,
                    <int> _num_classes,
                    <double*> _coef_ptr,
                    <double*> _pred_ptr)
            else:
                qnPredict[double, int](
                    handle_[0],
                    qnpams,
                    <double*><uintptr_t> X_m.ptr,
                    <bool> _is_col_major(X_m),
                    <int> n_rows,
                    <int> n_cols,
                    <int> _num_classes,
                    <double*> _coef_ptr,
                    <double*> _pred_ptr)

        self._calc_intercept()

        self.handle.sync()

        del X_m

        return preds

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def get_num_classes(self, _num_classes_dim):
        """
        Retrieves the number of classes from the classes dimension
        in the coefficients.
        """
        cdef qn_params qnpams = self.qnparams.params
        solves_classification = qnpams.loss in {
            qn_loss_type.QN_LOSS_LOGISTIC,
            qn_loss_type.QN_LOSS_SOFTMAX,
            qn_loss_type.QN_LOSS_SVC_L1,
            qn_loss_type.QN_LOSS_SVC_L2
        }
        solves_multiclass = qnpams.loss in {
            qn_loss_type.QN_LOSS_SOFTMAX
        }
        if solves_classification and not solves_multiclass:
            _num_classes = _num_classes_dim + 1
        else:
            _num_classes = _num_classes_dim
        return _num_classes

    def _calc_intercept(self):
        """
        If `fit_intercept == True`, then the last row of `coef_` contains
        `intercept_`. This should be called after every function that sets
        `coef_`
        """

        if self.fit_intercept:
            self.intercept_ = self._coef_[-1]
            return

        _num_classes_dim, _ = self.coef_.shape
        _num_classes = self.get_num_classes(_num_classes_dim)

        if _num_classes == 2:
            self.intercept_ = CumlArray.zeros(shape=1)
        else:
            self.intercept_ = CumlArray.zeros(shape=_num_classes)

    @classmethod
    def _get_param_names(cls):
        return super()._get_param_names() + \
            ['loss', 'fit_intercept', 'l1_strength', 'l2_strength',
                'max_iter', 'tol', 'linesearch_max_iter', 'lbfgs_memory',
                'warm_start', 'delta', 'penalty_normalized']


def _is_col_major(X):
    return getattr(X, "order", "F").upper() == "F"

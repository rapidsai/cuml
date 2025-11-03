# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import numpy as np
from pylibraft.common.handle import Handle

from cuml.common import input_to_cuml_array
from cuml.common.array_descriptor import CumlArrayDescriptor
from cuml.common.doc_utils import generate_docstring
from cuml.common.sparse_utils import is_sparse
from cuml.internals.array import CumlArray
from cuml.internals.array_sparse import SparseCumlArray
from cuml.internals.base import Base
from cuml.internals.logger import level_enum
from cuml.metrics import accuracy_score

from libc.stdint cimport uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t

from cuml.internals.logger cimport level_enum


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


SUPPORTED_LOSSES = {
    "sigmoid": qn_loss_type.QN_LOSS_LOGISTIC,
    "logistic": qn_loss_type.QN_LOSS_LOGISTIC,
    "softmax": qn_loss_type.QN_LOSS_SOFTMAX,
    "normal": qn_loss_type.QN_LOSS_SQUARED,
    "l2": qn_loss_type.QN_LOSS_SQUARED,
    "l1": qn_loss_type.QN_LOSS_ABS,
    "svc_l1": qn_loss_type.QN_LOSS_SVC_L1,
    "svc_l2": qn_loss_type.QN_LOSS_SVC_L2,
    "svr_l1": qn_loss_type.QN_LOSS_SVR_L1,
    "svr_l2": qn_loss_type.QN_LOSS_SVR_L2,
}


def fit_qn(
    X,
    y,
    sample_weight=None,
    *,
    convert_dtype=True,
    loss="sigmoid",
    bool fit_intercept=True,
    double l1_strength=0.0,
    double l2_strength=0.0,
    int max_iter=1000,
    double tol=1e-4,
    delta=None,
    int linesearch_max_iter=50,
    int lbfgs_memory=5,
    bool penalty_normalized=True,
    init_coef=None,
    level_enum verbose=level_enum.warn,
    handle=None,
):
    """Fit a linear model using a Quasi-newton method.

    Parameters
    ----------
    X : array-like, shape=(n_samples, n_features)
        The training data.
    y : array-like, shape=(n_samples,)
        The target values.
    sample_weight : None or array-like, shape=(n_samples,)
        The sample weights.
    convert_to_dtype : bool, default=True
        When set to True, will convert array inputs to be of the proper dtypes.
    **kwargs
        Remaining keyword arguments match the hyperparameters
        to ``QN``, see the ``QN`` docs for more information.

    Returns
    -------
    coef : CumlArray, shape=(1, n_features) or (n_classes, n_features)
        The fit coefficients.
    intercept : CumlArray, shape=(1,) or (n_classes,)
        Intercept added to the decision function.
    n_iter : int
        The number of iterations taken by the solver.
    """
    if handle is None:
        handle = Handle()

    cdef bool sparse_X = is_sparse(X)
    cdef int n_rows, n_cols, n_classes

    if sparse_X:
        X_m = SparseCumlArray(X, convert_index=np.int32)
        n_rows, n_cols = X_m.shape
        dtype = X_m.dtype
    else:
        X_m, n_rows, n_cols, dtype = input_to_cuml_array(
            X,
            convert_to_dtype=(np.float32 if convert_dtype else None),
            check_dtype=[np.float32, np.float64],
            order="K",
        )

    y_m = input_to_cuml_array(
        y,
        check_dtype=dtype,
        convert_to_dtype=(dtype if convert_dtype else None),
        check_rows=n_rows,
        check_cols=1,
    ).array

    if sample_weight is not None:
        sample_weight = input_to_cuml_array(
            sample_weight,
            check_dtype=dtype,
            check_rows=n_rows,
            check_cols=1,
            convert_to_dtype=(dtype if convert_dtype else None)
        )

    cdef qn_params params
    if (loss_type := SUPPORTED_LOSSES.get(loss)) is None:
        raise ValueError(f"{loss=!r} is unsupported")
    params.loss = loss_type
    params.penalty_l1 = l1_strength
    params.penalty_l2 = l2_strength
    params.grad_tol = tol
    params.change_tol = delta if delta is not None else tol * 0.01
    params.max_iter = max_iter
    params.linesearch_max_iter = linesearch_max_iter
    params.lbfgs_memory = lbfgs_memory
    params.verbose = <int>verbose
    params.fit_intercept = fit_intercept
    params.penalty_normalized = penalty_normalized

    is_classification = params.loss in (
        qn_loss_type.QN_LOSS_LOGISTIC,
        qn_loss_type.QN_LOSS_SOFTMAX,
        qn_loss_type.QN_LOSS_SVC_L1,
        qn_loss_type.QN_LOSS_SVC_L2
    )
    is_multiclass = params.loss == qn_loss_type.QN_LOSS_SOFTMAX

    if is_classification:
        n_classes = len(cp.unique(y_m))
    else:
        n_classes = 1

    if not is_multiclass and n_classes > 2:
        raise ValueError(
            f"The selected solver ({loss}) does not support"
            f" more than 2 classes ({n_classes} discovered)."
        )
    elif is_multiclass and n_classes <= 2:
        raise ValueError(
            "Two classes or less cannot be trained with softmax (multinomial)."
        )

    coef_n_cols = max(n_classes - 1, 1)
    coef_n_rows = n_cols + 1 if fit_intercept else n_cols

    if init_coef is None:
        coef = CumlArray.zeros((coef_n_rows, coef_n_cols), dtype=dtype, order="C")
    else:
        coef = input_to_cuml_array(
            init_coef,
            order="C",
            check_dtype=dtype,
            convert_to_dtype=(dtype if convert_dtype else None),
            check_rows=coef_n_rows,
            check_cols=coef_n_cols,
        )

    cdef uintptr_t X_ptr, X_indices_ptr, X_indptr_ptr
    cdef int X_nnz
    cdef bool X_is_col_major
    if sparse_X:
        X_ptr = X_m.data.ptr
        X_indices_ptr = X_m.indices.ptr
        X_indptr_ptr = X_m.indptr.ptr
        X_nnz = X_m.nnz
    else:
        X_ptr = X_m.ptr
        X_is_col_major = X_m.order == "F"

    cdef uintptr_t y_ptr = y_m.ptr
    cdef uintptr_t coef_ptr = coef.ptr
    cdef uintptr_t sample_weight_ptr = (
        0 if sample_weight is None else sample_weight.ptr
    )
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
    cdef float objective_f32
    cdef double objective_f64
    cdef int n_iter
    cdef bool use_float32 = dtype == np.float32

    with nogil:
        if sparse_X:
            if use_float32:
                qnFitSparse[float, int](
                    handle_[0],
                    params,
                    <float*> X_ptr,
                    <int*> X_indices_ptr,
                    <int*> X_indptr_ptr,
                    X_nnz,
                    <float*> y_ptr,
                    n_rows,
                    n_cols,
                    n_classes,
                    <float*> coef_ptr,
                    &objective_f32,
                    &n_iter,
                    <float*> sample_weight_ptr
                )
            else:
                qnFitSparse[double, int](
                    handle_[0],
                    params,
                    <double*> X_ptr,
                    <int*> X_indices_ptr,
                    <int*> X_indptr_ptr,
                    X_nnz,
                    <double*> y_ptr,
                    n_rows,
                    n_cols,
                    n_classes,
                    <double*> coef_ptr,
                    &objective_f64,
                    &n_iter,
                    <double*> sample_weight_ptr
                )
        else:
            if use_float32:
                qnFit[float, int](
                    handle_[0],
                    params,
                    <float*> X_ptr,
                    X_is_col_major,
                    <float*> y_ptr,
                    n_rows,
                    n_cols,
                    n_classes,
                    <float*> coef_ptr,
                    &objective_f32,
                    &n_iter,
                    <float*> sample_weight_ptr
                )
            else:
                qnFit[double, int](
                    handle_[0],
                    params,
                    <double*> X_ptr,
                    X_is_col_major,
                    <double*> y_ptr,
                    n_rows,
                    n_cols,
                    n_classes,
                    <double*> coef_ptr,
                    &objective_f64,
                    &n_iter,
                    <double*> sample_weight_ptr
                )

    handle.sync()

    coef = coef.to_output("cupy")

    if fit_intercept:
        intercept = coef[-1]
        coef = CumlArray(data=coef[:-1].T)
    else:
        if n_classes == 2:
            intercept = CumlArray.zeros(shape=1)
        else:
            intercept = CumlArray.zeros(shape=n_classes)
        coef = CumlArray(data=coef.T)

    return coef, intercept, n_iter


class QN(Base):
    """
    Quasi-Newton methods are used to either find zeroes or local maxima
    and minima of functions, and used by this class to optimize a cost
    function.

    Two algorithms are implemented underneath cuML's QN class, and which one
    is executed depends on the following rule:

      * Orthant-Wise Limited Memory Quasi-Newton (OWL-QN) if there is l1
        regularization

      * Limited Memory BFGS (L-BFGS) otherwise.

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

    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.solvers import QN
    >>> X = cp.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    >>> y = cp.array([0.0, 0.0, 1.0, 1.0])
    >>> solver = QN().fit(X, y)
    """

    coef_ = CumlArrayDescriptor()
    intercept_ = CumlArrayDescriptor()

    @classmethod
    def _get_param_names(cls):
        return [
            *super()._get_param_names(),
            "loss",
            "fit_intercept",
            "l1_strength",
            "l2_strength",
            "max_iter",
            "tol",
            "linesearch_max_iter",
            "lbfgs_memory",
            "warm_start",
            "delta",
            "penalty_normalized"
        ]

    def __init__(
        self,
        *,
        loss="sigmoid",
        fit_intercept=True,
        l1_strength=0.0,
        l2_strength=0.0,
        max_iter=1000,
        tol=1e-4,
        delta=None,
        linesearch_max_iter=50,
        lbfgs_memory=5,
        verbose=False,
        handle=None,
        output_type=None,
        warm_start=False,
        penalty_normalized=True,
    ):
        super().__init__(handle=handle, verbose=verbose, output_type=output_type)

        self.loss = loss
        self.fit_intercept = fit_intercept
        self.l1_strength = l1_strength
        self.l2_strength = l2_strength
        self.max_iter = max_iter
        self.tol = tol
        self.delta = delta
        self.linesearch_max_iter = linesearch_max_iter
        self.lbfgs_memory = lbfgs_memory
        self.warm_start = warm_start
        self.penalty_normalized = penalty_normalized

    @generate_docstring(X="dense_sparse")
    def fit(self, X, y, sample_weight=None, convert_dtype=True) -> "QN":
        """
        Fit the model with X and y.
        """
        # TODO: warm_start
        coef, intercept, n_iter = fit_qn(
            X,
            y,
            sample_weight=sample_weight,
            convert_dtype=convert_dtype,
            loss=self.loss,
            fit_intercept=self.fit_intercept,
            l1_strength=self.l1_strength,
            l2_strength=self.l2_strength,
            max_iter=self.max_iter,
            tol=self.tol,
            delta=self.delta,
            linesearch_max_iter=self.linesearch_max_iter,
            lbfgs_memory=self.lbfgs_memory,
            penalty_normalized=self.penalty_normalized,
            init_coef=None,
            verbose=self.verbose,
            handle=self.handle,
        )
        self.coef_ = coef
        self.intercept_ = intercept
        self.n_iter_ = n_iter

        return self

    @generate_docstring(X="dense_sparse")
    def predict(self, X) -> CumlArray:
        """Predicts the y for X."""
        # TODO
        raise NotImplementedError

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

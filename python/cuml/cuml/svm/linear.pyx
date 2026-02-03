# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import numpy as np

from cuml.internals.array import CumlArray
from cuml.internals.base import get_handle

from libc.stdint cimport uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t

from cuml.internals.logger cimport level_enum

__all__ = ("fit", "compute_probabilities")


cdef extern from "cuml/svm/linear.hpp" namespace "ML::SVM::linear" nogil:

    cdef enum Penalty "ML::SVM::linear::Params::Penalty":
        L1 "ML::SVM::linear::Params::L1"
        L2 "ML::SVM::linear::Params::L2"

    cdef enum Loss "ML::SVM::linear::Params::Loss":
        HINGE "ML::SVM::linear::Params::HINGE"
        SQUARED_HINGE "ML::SVM::linear::Params::SQUARED_HINGE"
        EPSILON_INSENSITIVE "ML::SVM::linear::Params::EPSILON_INSENSITIVE"
        SQUARED_EPSILON_INSENSITIVE "ML::SVM::linear::Params::SQUARED_EPSILON_INSENSITIVE"

    cdef struct Params:
        Penalty penalty
        Loss loss
        bool fit_intercept
        bool penalized_intercept
        int max_iter
        int linesearch_max_iter
        int lbfgs_memory
        level_enum verbose
        double C
        double grad_tol
        double change_tol
        double epsilon

    cdef int cpp_fit "ML::SVM::linear::fit"[T](
        const handle_t& handle,
        const Params& params,
        const size_t nRows,
        const size_t nCols,
        const int nClasses,
        const T* classes,
        const T* X,
        const T* y,
        const T* sampleWeight,
        T* w,
        T* probScale,
    ) except +

    cdef void computeProbabilities[T](
        const handle_t& handle,
        const size_t nRows,
        const int nClasses,
        const T* probScale,
        T* scores,
        T* out) except +


def _check_array(name, arr, dtype=None, shape=None, order=None):
    """Perform sanity checks.

    User-facing checks should happen earlier, this is just to enforce invariants.
    """
    if dtype is not None and arr.dtype != dtype:
        raise RuntimeError(f"Expected `{name}` with {dtype=}, got {arr.dtype!r}")
    if shape is not None:
        n_cols = shape[1] if len(shape) == 2 else 1
        ok_rows = arr.shape[0] == shape[0]
        ok_cols = arr.ndim == 1 and n_cols == 1 or arr.shape[1] == n_cols
        if not (ok_rows and ok_cols):
            raise RuntimeError(f"Expected `{name}` with {shape=}, got {arr.shape!r}")
    if order is not None and arr.order != order:
        raise RuntimeError(f"Expected `{name}` with {order=}, got {arr.order!r}")


def fit(
    X,
    y,
    sample_weight,
    *,
    n_classes=None,
    n_streams=0,
    probability=False,
    penalty,
    loss,
    fit_intercept,
    penalized_intercept,
    max_iter,
    linesearch_max_iter,
    lbfgs_memory,
    C,
    tol,
    epsilon,
    level_enum verbose,
):
    """Perform a Linear SVR or SVC fit.

    Parameters
    ----------
    handle : pylibraft.common.Handle
        The handle to use.
    X : CumlArray, shape = (n_samples, n_features)
        Training vectors
    y : CumlArray, shape = (n_samples,)
        Target values or classes
    sample_weight : None or CumlArray, shape = (n_samples,), default=None
        Sample weights
    n_classes : int or None, default=None
        The number of classes, or None if fitting a regression problem.
    n_streams : int, default=0
        The number of streams to use when fitting classes.
    probability : bool, default=False
        When fitting an SVC, whether to also fit probability scales to enable
        `predict_proba`.
    **kwargs
        Remaining common hyperparameters for LinearSVC/LinearSVR, see their
        docstrings for details. These are required keyword-only to ensure
        they're properly plumbed through at all callsites.

    Returns
    -------
    coef_ : CumlArray, shape = (1, n_features) or (n_classes, n_features)
        The fitted coefficients. Has 1 row for regression and binary
        classification, or n_classes rows for multi-class classification.
    intercept_ : float or CumlArray, shape = (1,) or (n_classes,)
        The fitted intercept. If `fit_intercept=False`, returns 0.0 (matching
        sklearn behavior). Otherwise returns a 1-element array for regression
        and binary classification, or `n_classes` elements for multi-class
        classification.
    n_iter_ : int
        The maximum number of iterations run across all classes.
    prob_scale_ : None or CumlArray, shape = (n_classes, 2)
        The probability scales (if `probability=True`), `None` otherwise.
    """
    penalties = {"l1": Penalty.L1, "l2": Penalty.L2}
    if n_classes is not None:
        losses = {"hinge": Loss.HINGE, "squared_hinge": Loss.SQUARED_HINGE}
    else:
        losses = {
            "epsilon_insensitive": Loss.EPSILON_INSENSITIVE,
            "squared_epsilon_insensitive": Loss.SQUARED_EPSILON_INSENSITIVE,
        }

    # Process parameters
    cdef Params params

    if penalty not in penalties:
        raise ValueError(
            f"Expected penalty to be one of {list(penalties)}, got {penalty!r}"
        )
    else:
        params.penalty = penalties[penalty]

    if loss not in losses:
        raise ValueError(
            f"Expected loss to be one of {list(losses)}, got {loss!r}"
        )
    else:
        params.loss = losses[loss]

    params.fit_intercept = fit_intercept
    params.penalized_intercept = penalized_intercept
    params.max_iter = max_iter
    params.linesearch_max_iter = linesearch_max_iter
    params.lbfgs_memory = lbfgs_memory
    params.C = C
    params.epsilon = epsilon
    params.grad_tol = tol
    params.change_tol = 0.1 * tol
    params.verbose = verbose

    # Extract dimensions
    cdef size_t n_rows = X.shape[0]
    cdef size_t n_cols = X.shape[1]

    # Validate dimensions
    if n_rows < 1:
        raise ValueError(
            f"Found array with {n_rows} sample(s) (shape={X.shape}) while a "
            f"minimum of 1 is required."
        )
    if n_cols < 1:
        raise ValueError(
            f"Found array with {n_cols} feature(s) (shape={X.shape}) while a "
            f"minimum of 1 is required."
        )
    if n_classes == 1:
        raise ValueError(
            "This solver needs samples of at least 2 classes in the data, but "
            "the data contains only 1 class"
        )

    # Sanity checks
    _check_array("X", X, order="F")
    _check_array("y", y, dtype=X.dtype, shape=(n_rows,))
    if sample_weight is not None:
        _check_array("sample_weight", sample_weight, dtype=X.dtype, shape=(n_rows,))

    if n_classes is not None:
        classes = cp.arange(n_classes, dtype=X.dtype)

    # Allocate output arrays
    n_coefs = n_cols + int(fit_intercept)
    if n_classes is not None:
        w_shape = (1 if n_classes == 2 else n_classes, n_coefs)
    else:
        w_shape = n_coefs
    w = CumlArray.empty(shape=w_shape, dtype=X.dtype, order="F")

    if probability and n_classes is not None:
        prob_scale = CumlArray.empty((n_classes, 2), dtype=X.dtype, order="F")
    else:
        prob_scale = None

    handle = get_handle(n_streams=n_streams)
    cdef handle_t *handle_ = <handle_t*><size_t>handle.getHandle()
    cdef bool is_float32 = X.dtype == np.float32
    cdef int n_classes_or_0 = 0 if n_classes is None else n_classes
    cdef uintptr_t X_ptr = X.ptr
    cdef uintptr_t y_ptr = y.ptr
    cdef uintptr_t sample_weight_ptr = 0 if sample_weight is None else sample_weight.ptr
    cdef uintptr_t classes_ptr = 0 if n_classes is None else classes.data.ptr
    cdef uintptr_t w_ptr = w.ptr
    cdef uintptr_t prob_scale_ptr = 0 if prob_scale is None else prob_scale.ptr
    cdef int n_iter

    # Perform fit
    with nogil:
        if is_float32:
            n_iter = cpp_fit[float](
                handle_[0],
                params,
                n_rows,
                n_cols,
                n_classes_or_0,
                <const float*>classes_ptr,
                <const float*>X_ptr,
                <const float*>y_ptr,
                <const float*>sample_weight_ptr,
                <float*>w_ptr,
                <float*>prob_scale_ptr,
            )
        else:
            n_iter = cpp_fit[double](
                handle_[0],
                params,
                n_rows,
                n_cols,
                n_classes_or_0,
                <const double*>classes_ptr,
                <const double*>X_ptr,
                <const double*>y_ptr,
                <const double*>sample_weight_ptr,
                <double*>w_ptr,
                <double*>prob_scale_ptr,
            )
    handle.sync()

    # Decompose `w` into coef and intercept
    if fit_intercept:
        if w.ndim == 2:
            coef = w[:, :-1]
            intercept = CumlArray(data=w.to_output("cupy")[:, -1:].flatten())
        else:
            coef = w[:-1]
            intercept = w[-1:]
    else:
        coef = w
        intercept = 0.0

    return coef, intercept, n_iter, prob_scale


def compute_probabilities(scores, prob_scale, n_streams):
    """Compute probabilities from decision function scores.

    Parameters
    ----------
    scores : CumlArray, shape = (n_samples, n_classes)
        The decision function scores.
    prob_scale : CumlArray, shape = (n_classes, 2)
        The probability scaling factors.
    n_streams : int
        The number of streams to use.

    Returns
    -------
    probabilities : CumlArray, shape = (n_samples, n_classes)
        The computed probabilities.
    """
    # Extract dimensions
    cdef size_t n_rows = scores.shape[0]
    cdef int n_classes = prob_scale.shape[0]

    # Sanity checks
    _check_array("scores", scores, order="C")
    _check_array(
        "prob_scale", prob_scale, dtype=scores.dtype, order="F", shape=(n_classes, 2)
    )

    # Allocate outputs
    out = CumlArray.empty((n_rows, n_classes), dtype=scores.dtype, order="C")

    handle = get_handle(n_streams=n_streams)
    cdef handle_t *handle_ = <handle_t*><size_t>handle.getHandle()
    cdef bool is_float32 = scores.dtype == np.float32
    cdef uintptr_t scores_ptr = scores.ptr
    cdef uintptr_t prob_scale_ptr = prob_scale.ptr
    cdef uintptr_t out_ptr = out.ptr

    # Compute probabilities
    with nogil:
        if is_float32:
            computeProbabilities[float](
                handle_[0],
                n_rows,
                n_classes,
                <const float*>prob_scale_ptr,
                <float*>scores_ptr,
                <float*>out_ptr
            )
        else:
            computeProbabilities[double](
                handle_[0],
                n_rows,
                n_classes,
                <const double*>prob_scale_ptr,
                <double*>scores_ptr,
                <double*>out_ptr
            )
    handle.sync()
    return out

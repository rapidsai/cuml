#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
import cupy as cp
import numpy as np

import cuml.internals.nvtx as nvtx
from cuml.common import CumlArray
from cuml.internals.base import get_handle

from libc.stdint cimport uint8_t, uintptr_t
from libcpp cimport bool
from pylibraft.common.handle cimport handle_t

__all__ = (
    "nnls",
    "nnls_batched",
    "fit_nnls_apg",
    "fit_nnls_batched",
)

_NVTX_DOMAIN = "cuml_python"
_NVTX_CATEGORY = "solvers.nnls"

_VALID_SOLVERS = {"cd", "sgd", "lbfgs", "lawson", "lawson_multikernel", "apg"}
_SUPPORTED_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))


cdef extern from "cuml/solvers/nnls.hpp" namespace "ML::Solver" nogil:
    cdef enum class NnlsLipschitzMethod(int):
        POWER_ITERATION = 0
        SVD = 1
        USER_SUPPLIED = 2

    cdef cppclass NnlsApgParams:
        int max_iter
        double tol
        int check_every
        bool restart
        NnlsLipschitzMethod lipschitz_method
        double lipschitz_value
        int power_iter
        double lipschitz_safety
        NnlsApgParams() except +

    cdef int nnlsApg(
        handle_t& handle,
        const float* A,
        int n_rows,
        int n_cols,
        const float* b,
        float* x,
        const NnlsApgParams& params,
    ) except +

    cdef int nnlsApg(
        handle_t& handle,
        const double* A,
        int n_rows,
        int n_cols,
        const double* b,
        double* x,
        const NnlsApgParams& params,
    ) except +

    cdef enum class NnlsBatchedSolver(int):
        LAWSON = 0
        APG = 1
        CD = 2
        SGD = 3
        LBFGS = 4
        LAWSON_MULTIKERNEL = 5

    cdef cppclass NnlsBatchedParams:
        NnlsBatchedSolver solver
        int max_iter
        double tol
        int check_every
        int lbfgs_history
        NnlsLipschitzMethod lipschitz_method
        double lipschitz_value
        int power_iter
        double lipschitz_safety
        NnlsBatchedParams() except +

    cdef void nnlsBatched(
        handle_t& handle,
        const float* A,
        int m,
        int n,
        const float* B,
        int n_problems,
        const uint8_t* masks,
        float* X,
        float* fitted,
        const NnlsBatchedParams& params,
    ) except +

    cdef void nnlsBatched(
        handle_t& handle,
        const double* A,
        int m,
        int n,
        const double* B,
        int n_problems,
        const uint8_t* masks,
        double* X,
        double* fitted,
        const NnlsBatchedParams& params,
    ) except +


_LIPSCHITZ_METHODS = {
    "power": NnlsLipschitzMethod.POWER_ITERATION,
    "svd": NnlsLipschitzMethod.SVD,
    "user": NnlsLipschitzMethod.USER_SUPPLIED,
}

_BATCHED_SOLVERS = {
    "lawson": NnlsBatchedSolver.LAWSON,
    "lawson_multikernel": NnlsBatchedSolver.LAWSON_MULTIKERNEL,
    "apg": NnlsBatchedSolver.APG,
    "cd": NnlsBatchedSolver.CD,
    "sgd": NnlsBatchedSolver.SGD,
    "lbfgs": NnlsBatchedSolver.LBFGS,
}


def fit_nnls_apg(
    A,
    b,
    *,
    convert_dtype=False,
    int max_iter=1000,
    double tol=1e-6,
    int check_every=10,
    bint restart=True,
    str lipschitz="power",
    double lipschitz_value=0.0,
    int power_iter=30,
    double lipschitz_safety=1.05,
):
    """Solve a Non-Negative Least Squares problem with FISTA-style accelerated
    projected gradient (APG).

    The smooth quadratic objective ``f(x) = 1/2 || A x - b ||_2^2`` is
    minimized over ``x >= 0`` by repeatedly taking a gradient step ``y - g/L``
    (with ``g = A^T (A y - b)``), projecting onto the nonnegative orthant,
    and adding Nesterov momentum.  Each iteration costs two ``gemv`` calls
    plus a single fused projection / momentum elementwise pass; the Hessian
    ``A^T A`` is never materialised.

    Parameters
    ----------
    A : array-like, shape=(n_rows, n_cols)
        Coefficient matrix, dtype ``float32`` or ``float64``.
    b : array-like, shape=(n_rows,)
        Right-hand-side vector.  Cast to ``A.dtype`` if needed (and if
        ``convert_dtype=True``; otherwise its dtype must already match).
    convert_dtype : bool, default=False
        If True, convert ``A`` and ``b`` to a supported dtype.
    max_iter : int, default=1000
        Maximum number of APG iterations.
    tol : double, default=1e-6
        KKT tolerance.  The solver stops when
        ``max_j |min(x_j, g_j)| < tol * max(1, ||A^T b||_inf)``.
    check_every : int, default=10
        Number of iterations between KKT-residual checks (and therefore
        between device->host syncs).
    restart : bool, default=True
        Enable adaptive (gradient) restart of the Nesterov momentum.
    lipschitz : {'power', 'svd', 'user'}, default='power'
        Method for estimating the gradient Lipschitz constant
        ``L = sigma_max(A) ** 2`` that controls the step size ``1/L``.

        - ``'power'`` -- a few power iterations on ``A^T A`` (default,
          two ``gemv`` per step, no extra factorization).
        - ``'svd'``   -- exact ``L`` via cuSOLVER's Jacobi SVD on a copy
          of ``A`` (costlier, intended for small matrices).
        - ``'user'``  -- use ``lipschitz_value`` directly.
    lipschitz_value : double, default=0.0
        Explicit Lipschitz constant.  Only used when ``lipschitz='user'``;
        must be strictly positive in that case.
    power_iter : int, default=30
        Number of power iterations performed when ``lipschitz='power'``.
    lipschitz_safety : double, default=1.05
        Multiplicative safety factor on the estimated ``L`` (estimates can
        slightly underestimate ``sigma_max`` and the projected step requires
        ``1/L <= 1/||A||_2^2``).

    Returns
    -------
    coef : CumlArray, shape=(n_cols,)
        The non-negative solution vector.
    """
    if lipschitz not in _LIPSCHITZ_METHODS:
        raise ValueError(
            f"Unknown lipschitz method {lipschitz!r}. "
            f"Expected one of {sorted(_LIPSCHITZ_METHODS)}."
        )
    if lipschitz == "user" and not (lipschitz_value > 0):
        raise ValueError(
            "lipschitz='user' requires lipschitz_value > 0; got "
            f"{lipschitz_value!r}."
        )
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}.")
    if check_every < 1:
        raise ValueError(f"check_every must be >= 1, got {check_every}.")

    handle = get_handle()

    cdef int n_rows, n_cols
    A = CumlArray.from_input(
        A,
        check_dtype=[np.float32, np.float64],
        convert_to_dtype=(np.float32 if convert_dtype else None),
        order="F",
    )
    n_rows = A.shape[0]
    n_cols = A.shape[1] if len(A.shape) > 1 else 1

    if n_rows < 1:
        raise ValueError(
            f"Found array with {n_rows} sample(s) (shape={A.shape}) while a "
            f"minimum of 1 is required."
        )
    if n_cols < 1:
        raise ValueError(
            f"Found array with {n_cols} feature(s) (shape={A.shape}) while "
            f"a minimum of 1 is required."
        )

    b = CumlArray.from_input(
        b,
        check_dtype=A.dtype,
        convert_to_dtype=(A.dtype if convert_dtype else None),
        check_rows=n_rows,
        check_cols=1,
    )

    coef = CumlArray(cp.zeros(n_cols, dtype=A.dtype))

    cdef NnlsApgParams params
    params.max_iter         = max_iter
    params.tol              = tol
    params.check_every      = check_every
    params.restart          = restart
    params.lipschitz_method = _LIPSCHITZ_METHODS[lipschitz]
    params.lipschitz_value  = lipschitz_value
    params.power_iter       = power_iter
    params.lipschitz_safety = lipschitz_safety

    cdef uintptr_t A_ptr = A.ptr
    cdef uintptr_t b_ptr = b.ptr
    cdef uintptr_t coef_ptr = coef.ptr
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
    cdef bint is_float32 = A.dtype == np.float32

    with nogil:
        if is_float32:
            nnlsApg(
                handle_[0],
                <const float*>A_ptr,
                n_rows,
                n_cols,
                <const float*>b_ptr,
                <float*>coef_ptr,
                params,
            )
        else:
            nnlsApg(
                handle_[0],
                <const double*>A_ptr,
                n_rows,
                n_cols,
                <const double*>b_ptr,
                <double*>coef_ptr,
                params,
            )
    handle.sync()

    return coef


def fit_nnls_batched(
    A,
    B,
    masks=None,
    *,
    convert_dtype=False,
    str solver="lawson",
    int max_iter=0,
    double tol=1e-6,
    int check_every=10,
    int lbfgs_history=5,
    str lipschitz="power",
    double lipschitz_value=0.0,
    int power_iter=30,
    double lipschitz_safety=1.05,
    bint compute_fitted=True,
):
    """Solve a batch of masked, shared-``A`` Non-Negative Least Squares
    problems in a single kernel launch per backend.

    For every column ``p`` of ``B`` this solves
    ``argmin_{x >= 0, x[j]=0 where masks[j, p]==0} || A @ x - B[:, p] ||_2``
    where ``A`` is shared across the whole batch. The Gram matrix
    ``G = A.T @ A`` and the projections ``C = A.T @ B`` are formed once and
    reused by every problem.

    Parameters
    ----------
    A : array-like, shape=(m, n)
        Shared coefficient matrix, dtype ``float32`` or ``float64``.
    B : array-like, shape=(m, n_problems)
        Right-hand-side matrix (one column per problem).
    masks : array-like, shape=(n, n_problems), optional
        Column-major (``(n_signatures, n_problems)``) boolean/uint8 support;
        ``masks[j, p]`` is nonzero iff column ``j`` is active for problem ``p``.
        Pass an **F-contiguous** array for a zero-copy device path: its memory
        layout (signature index contiguous) then matches the kernel's per-block
        access exactly. ``None`` means every column is active for every problem.
    convert_dtype : bool, default=False
        If True, convert ``A``/``B`` to a supported dtype.
    solver : {'lawson', 'apg', 'cd', 'sgd', 'lbfgs'}, default='lawson'
        Backend to use. ``'lawson'`` is the exact active-set method and the
        best fit for small ``n``; the others minimise the same QP iteratively.
    max_iter : int, default=0
        Iteration cap. ``0`` selects the per-backend default.
    tol : double, default=1e-6
        Relative KKT tolerance.
    check_every : int, default=10
        Iterations between convergence checks (iterative backends).
    lbfgs_history : int, default=5
        History length for the LBFGS backend.
    lipschitz, lipschitz_value, power_iter, lipschitz_safety
        Options for the gradient backends' step size ``1/L`` (see
        :func:`fit_nnls_apg`). Ignored by ``lawson`` and ``cd``.
    compute_fitted : bool, default=True
        Whether to also return ``fitted = A @ X``.

    Returns
    -------
    X : CumlArray, shape=(n, n_problems)
        Non-negative solutions (column-major), masked-out rows set to 0.
    fitted : CumlArray, shape=(m, n_problems) or None
        ``A @ X`` when ``compute_fitted`` else ``None``.
    """
    if solver not in _BATCHED_SOLVERS:
        raise ValueError(
            f"Unknown solver {solver!r}. "
            f"Expected one of {sorted(_BATCHED_SOLVERS)}."
        )
    if lipschitz not in _LIPSCHITZ_METHODS:
        raise ValueError(
            f"Unknown lipschitz method {lipschitz!r}. "
            f"Expected one of {sorted(_LIPSCHITZ_METHODS)}."
        )

    handle = get_handle()

    cdef int m, n
    A = CumlArray.from_input(
        A,
        check_dtype=[np.float32, np.float64],
        convert_to_dtype=(np.float32 if convert_dtype else None),
        order="F",
    )
    m = A.shape[0]
    n = A.shape[1] if len(A.shape) > 1 else 1

    if m < 1:
        raise ValueError(
            f"Found array with {m} sample(s) (shape={A.shape}) while a "
            f"minimum of 1 is required."
        )
    if n < 1:
        raise ValueError(
            f"Found array with {n} feature(s) (shape={A.shape}) while "
            f"a minimum of 1 is required."
        )

    cdef int n_problems
    B = CumlArray.from_input(
        B,
        check_dtype=A.dtype,
        convert_to_dtype=(A.dtype if convert_dtype else None),
        check_rows=m,
        order="F",
    )
    n_problems = B.shape[1] if len(B.shape) > 1 else 1

    cdef uintptr_t masks_ptr = 0
    if masks is not None:
        # Column-major (n, n_problems): F-contiguous input is used in place, and
        # its raw layout (signature index fastest) matches the kernel's
        # ``masks[p*n + j]`` per-block access. A C-contiguous input is copied to
        # this layout by ``from_input`` (still correct, just not zero-copy).
        masks_arr = CumlArray.from_input(
            masks,
            check_dtype=[np.uint8],
            convert_to_dtype=np.uint8,
            check_rows=n,
            check_cols=n_problems,
            order="F",
        )
        masks_ptr = masks_arr.ptr

    X = CumlArray(cp.zeros((n, n_problems), dtype=A.dtype, order="F"))

    fitted = None
    cdef uintptr_t fitted_ptr = 0
    if compute_fitted:
        fitted = CumlArray(cp.zeros((m, n_problems), dtype=A.dtype, order="F"))
        fitted_ptr = fitted.ptr

    cdef NnlsBatchedParams params
    params.solver           = _BATCHED_SOLVERS[solver]
    params.max_iter         = max_iter
    params.tol              = tol
    params.check_every      = check_every
    params.lbfgs_history    = lbfgs_history
    params.lipschitz_method = _LIPSCHITZ_METHODS[lipschitz]
    params.lipschitz_value  = lipschitz_value
    params.power_iter       = power_iter
    params.lipschitz_safety = lipschitz_safety

    cdef uintptr_t A_ptr = A.ptr
    cdef uintptr_t B_ptr = B.ptr
    cdef uintptr_t X_ptr = X.ptr
    cdef handle_t* handle_ = <handle_t*><size_t>handle.getHandle()
    cdef bint is_float32 = A.dtype == np.float32

    with nogil:
        if is_float32:
            nnlsBatched(
                handle_[0],
                <const float*>A_ptr,
                m,
                n,
                <const float*>B_ptr,
                n_problems,
                <const uint8_t*>masks_ptr,
                <float*>X_ptr,
                <float*>fitted_ptr,
                params,
            )
        else:
            nnlsBatched(
                handle_[0],
                <const double*>A_ptr,
                m,
                n,
                <const double*>B_ptr,
                n_problems,
                <const uint8_t*>masks_ptr,
                <double*>X_ptr,
                <double*>fitted_ptr,
                params,
            )
    handle.sync()

    return X, fitted


def nnls(
    A,
    b,
    *,
    maxiter=None,
    solver="cd",
    compute_rnorm=True,
    tol=None,
    check_every=10,
    restart=True,
    lipschitz="power",
):
    """Solve ``argmin_x || A @ x - b ||_2`` for ``x >= 0``.

    This is a GPU-accelerated equivalent of :func:`scipy.optimize.nnls`.

    Parameters
    ----------
    A : array-like, shape (m, n)
        Coefficient matrix.  Accepts NumPy arrays, CuPy arrays, or any
        ``__cuda_array_interface__`` object.  Must have dtype ``float32`` or
        ``float64``; other dtypes are cast to ``float32``.
    b : array-like, shape (m,)
        Right-hand side vector.  Will be cast to ``A.dtype`` if needed.
    maxiter : int, optional
        Maximum number of iterations.  Defaults to ``1000``.
    solver : {'cd', 'sgd', 'lbfgs', 'lawson', 'apg'}, default='cd'
        Which solver backend to use:

        - ``'cd'``     -- Coordinate Descent (default, best general-purpose).
        - ``'sgd'``    -- Projected Stochastic Gradient Descent.
        - ``'lbfgs'``  -- Projected L-BFGS (quasi-Newton).
        - ``'lawson'`` -- Single-kernel Lawson-Hanson active-set solver.
          Solves the whole problem in a single CUDA kernel using normal
          equations and shared-memory Cholesky.  Best for small problems
          (n_cols up to roughly 90 in double precision) where launch
          latency dominates the iterative solvers.
        - ``'apg'``    -- FISTA-style accelerated projected gradient on the
          QP form ``min 1/2 x^T (A^T A) x - (A^T b)^T x  s.t. x >= 0``.
          Built exclusively from raft / cuBLAS / cuSOLVER primitives and
          scales well to large, wide, or many-RHS workloads where the
          coordinate-descent inner loop is dominated by per-coefficient
          launch overhead.
    compute_rnorm : bool, default=True
        Whether to compute the residual 2-norm.  When ``False``, ``rnorm``
        is returned as ``None`` and an extra matmul + reduction + device sync
        per call is avoided.  Useful in tight loops where the caller does
        not need the residual norm.
    tol, check_every, restart, lipschitz
        APG-specific options (see :func:`fit_nnls_apg`).
        Ignored for the other solvers.  ``tol`` defaults to ``1e-6`` for APG.

    Returns
    -------
    x : cupy.ndarray, shape (n,)
        Solution vector with all entries >= 0.
    rnorm : float or None
        The 2-norm of the residual, ``|| A @ x - b ||_2``, or ``None`` when
        ``compute_rnorm=False``.

    Examples
    --------
    >>> import cupy as cp
    >>> from cuml.solvers import nnls
    >>> A = cp.array([[1, 0], [1, 0], [0, 1]], dtype=cp.float32)
    >>> b = cp.array([2, 1, 1], dtype=cp.float32)
    >>> x, rnorm = nnls(A, b)
    """
    if solver not in _VALID_SOLVERS:
        raise ValueError(
            f"Unknown solver {solver!r}. Expected one of {sorted(_VALID_SOLVERS)}"
        )

    with nvtx.annotate(
        message=f"nnls[{solver}]",
        domain=_NVTX_DOMAIN,
        category=_NVTX_CATEGORY,
    ):
        with nvtx.annotate(
            message="nnls.prepare_inputs",
            domain=_NVTX_DOMAIN,
            category=_NVTX_CATEGORY,
        ):
            A_gpu = cp.asarray(A)
            b_gpu = cp.asarray(b)

            if A_gpu.ndim != 2:
                raise ValueError(f"Expected 2-D array for A, got {A_gpu.ndim}-D")
            if b_gpu.ndim != 1:
                raise ValueError(f"Expected 1-D array for b, got {b_gpu.ndim}-D")
            if A_gpu.shape[0] != b_gpu.shape[0]:
                raise ValueError(
                    f"Incompatible dimensions: A has {A_gpu.shape[0]} rows, "
                    f"b has {b_gpu.shape[0]} elements"
                )

            # Ensure dtype is supported by the underlying solvers without
            # forcing an unconditional float64 -> float32 conversion (which
            # costs a kernel plus a device sync per call inside
            # ``input_to_cuml_array`` whenever ``convert_dtype=True``).
            # By doing the cast explicitly here we can then pass
            # ``convert_dtype=False`` and skip that overhead entirely.
            if A_gpu.dtype not in _SUPPORTED_DTYPES:
                A_gpu = A_gpu.astype(np.float32, copy=False)
            if b_gpu.dtype != A_gpu.dtype:
                b_gpu = b_gpu.astype(A_gpu.dtype, copy=False)

        max_iter = maxiter if maxiter is not None else 1000

        with nvtx.annotate(
            message=f"nnls.solve[{solver}]",
            domain=_NVTX_DOMAIN,
            category=_NVTX_CATEGORY,
        ):
            if solver in ("cd", "sgd", "lbfgs", "lawson", "lawson_multikernel"):
                # These backends are all implemented by the batched Gram-form
                # engine (fit_nnls_batched); a single-RHS problem is just a
                # batch of one column with no mask, so we reuse that path here
                # rather than maintaining a parallel single-problem kernel.
                # For the Lawson backends a max_iter of 0 selects the tight
                # active-set cap (3 * n_cols + 1) and the tolerance default is
                # the active-set gradient tolerance (1e-4) rather than 1e-6.
                default_tol = (
                    1e-4 if solver in ("lawson", "lawson_multikernel") else 1e-6
                )
                X_batched, _ = fit_nnls_batched(
                    A_gpu,
                    b_gpu.reshape(-1, 1),
                    None,
                    convert_dtype=False,
                    solver=solver,
                    max_iter=(0 if maxiter is None else max_iter),
                    tol=(default_tol if tol is None else float(tol)),
                    check_every=int(check_every),
                    lbfgs_history=5,
                    lipschitz=str(lipschitz),
                    compute_fitted=False,
                )
                coef = cp.asarray(X_batched)[:, 0]
            elif solver == "apg":
                coef = fit_nnls_apg(
                    A_gpu,
                    b_gpu,
                    convert_dtype=False,
                    max_iter=max_iter,
                    tol=(1e-6 if tol is None else float(tol)),
                    check_every=int(check_every),
                    restart=restart,
                    lipschitz=str(lipschitz),
                )

        x = cp.asarray(coef).ravel()
        if compute_rnorm:
            with nvtx.annotate(
                message="nnls.compute_rnorm",
                domain=_NVTX_DOMAIN,
                category=_NVTX_CATEGORY,
            ):
                residual = A_gpu @ x - b_gpu
                rnorm = float(cp.linalg.norm(residual))
        else:
            rnorm = None

        return x, rnorm


def nnls_batched(
    A,
    B,
    masks=None,
    b_index=None,
    *,
    solver="lawson",
    compute_fitted=True,
    maxiter=None,
    tol=1e-6,
    check_every=10,
    lbfgs_history=5,
    lipschitz="power",
):
    """Solve a batch of masked, shared-``A`` NNLS problems on the GPU.

    For every problem ``j`` this solves::

        argmin_x  || A[:, masks[:, j]] @ x - B[:, b_index[j]] ||_2
        subject to  x >= 0

    The design matrix ``A`` is shared across the whole batch; only the active
    column set (``masks``) and the target (``b_index``) vary per problem.

    This is a device-native cuML primitive: it returns device (cupy) arrays and
    keeps everything on the GPU.  Passing cupy inputs avoids any host transfer,
    so a caller can keep ``A``/``B`` resident across repeated calls simply by
    holding on to cupy arrays.

    Parameters
    ----------
    A : array-like, shape (m, n)
        Shared coefficient matrix (signatures).  NumPy or cupy; cupy inputs are
        used in place (no copy) when already ``float32``/``float64``.
    B : array-like, shape (m, n_targets)
        Distinct target vectors.  Not duplicated per problem; ``b_index``
        selects which column of ``B`` each problem uses.
    masks : array-like, shape (n, n_problems), bool/uint8, optional
        Column ``j`` selects the active columns of ``A`` for problem ``j``
        (``masks[i, j]`` nonzero iff signature ``i`` is active for problem
        ``j``).  For a zero-copy device path this should be an **F-contiguous**
        ``(n, n_problems)`` array.  ``None`` means every column is active for
        every problem (``n_problems == n_targets``).
    b_index : array-like, shape (n_problems,), int, optional
        Target column of ``B`` for each problem, gathered on the device.
        Defaults to the identity mapping (requires ``n_targets == n_problems``).
    solver : {'lawson', 'apg', 'cd', 'sgd', 'lbfgs'}, default='lawson'
        Backend to use.
    compute_fitted : bool, default=True
        Whether to also return ``fitted = A @ X``.  When ``False``, ``fitted``
        is ``None`` and the extra matmul is skipped.
    maxiter : int, optional
        Iteration cap.  Defaults to the per-backend default (selected by 0).
    tol : float, default=1e-6
        Relative KKT tolerance.
    check_every : int, default=10
        Iterations between convergence checks (iterative backends).
    lbfgs_history : int, default=5
        History length for the ``'lbfgs'`` backend.
    lipschitz : {'power', 'svd', 'user'}, default='power'
        Lipschitz-estimation method for the gradient backends' ``1/L`` step.
        Ignored by ``'lawson'`` and ``'cd'``.

    Returns
    -------
    X : cupy.ndarray, shape (n, n_problems)
        Non-negative solutions (column-major), masked-out rows set to 0.
    fitted : cupy.ndarray, shape (m, n_problems) or None
        ``A @ X`` per problem when ``compute_fitted``, else ``None``.
    """
    if solver not in _VALID_SOLVERS:
        raise ValueError(
            f"Unknown solver {solver!r}. Expected one of {sorted(_VALID_SOLVERS)}"
        )

    with nvtx.annotate(
        message=f"nnls_batched[{solver}]",
        domain=_NVTX_DOMAIN,
        category=_NVTX_CATEGORY,
    ):
        with nvtx.annotate(
            message="nnls_batched.prepare_inputs",
            domain=_NVTX_DOMAIN,
            category=_NVTX_CATEGORY,
        ):
            A_gpu = cp.asarray(A)
            B_gpu = cp.asarray(B)

            if A_gpu.ndim != 2:
                raise ValueError(f"Expected 2-D array for A, got {A_gpu.ndim}-D")
            if B_gpu.ndim != 2:
                raise ValueError(f"Expected 2-D array for B, got {B_gpu.ndim}-D")
            if A_gpu.shape[0] != B_gpu.shape[0]:
                raise ValueError(
                    f"Incompatible dimensions: A has {A_gpu.shape[0]} rows, "
                    f"B has {B_gpu.shape[0]} rows"
                )

            # Cast to a supported dtype once, up front, so the Cython layer can
            # run with convert_dtype=False (no extra kernel + sync per call).
            if A_gpu.dtype not in _SUPPORTED_DTYPES:
                A_gpu = A_gpu.astype(np.float32, copy=False)
            if B_gpu.dtype != A_gpu.dtype:
                B_gpu = B_gpu.astype(A_gpu.dtype, copy=False)

            n_cols = A_gpu.shape[1]

            masks_gpu = None
            n_problems = None
            if masks is not None:
                masks_gpu = cp.asarray(masks)
                if masks_gpu.ndim != 2:
                    raise ValueError(
                        f"Expected 2-D masks, got {masks_gpu.ndim}-D"
                    )
                if masks_gpu.shape[0] != n_cols:
                    raise ValueError(
                        f"Expected masks with {n_cols} rows (signatures), "
                        f"got {masks_gpu.shape[0]}"
                    )
                n_problems = masks_gpu.shape[1]
                # uint8, F-contiguous (n, n_problems): a no-op when the caller
                # already supplies that layout, so the Cython/kernel path is
                # zero-copy and the mask reads are coalesced.
                masks_gpu = cp.asfortranarray(
                    masks_gpu.astype(np.uint8, copy=False)
                )

            # Per-problem RHS gather on the device (no host copy, no duplication
            # of distinct targets on the host side).
            if b_index is not None:
                b_index_gpu = (
                    cp.asarray(b_index).astype(cp.int64, copy=False).ravel()
                )
                n_targets = B_gpu.shape[1]
                if b_index_gpu.size and (
                    int(b_index_gpu.max()) >= n_targets
                    or int(b_index_gpu.min()) < 0
                ):
                    raise ValueError(
                        f"b_index entries must lie in [0, {n_targets})"
                    )
                if n_problems is not None and b_index_gpu.shape[0] != n_problems:
                    raise ValueError(
                        f"Expected b_index of length {n_problems}, "
                        f"got {b_index_gpu.shape[0]}"
                    )
                B_gpu = B_gpu[:, b_index_gpu]

            if n_problems is None:
                n_problems = B_gpu.shape[1]
            elif B_gpu.shape[1] != n_problems:
                raise ValueError(
                    f"masks has {n_problems} problems but B (after b_index "
                    f"gather) has {B_gpu.shape[1]} columns"
                )

            # The batched kernel consumes column-major (F) B.
            B_gpu = cp.asfortranarray(B_gpu)

        X, fitted = fit_nnls_batched(
            A_gpu,
            B_gpu,
            masks_gpu,
            convert_dtype=False,
            solver=str(solver),
            max_iter=(0 if maxiter is None else int(maxiter)),
            tol=(1e-6 if tol is None else float(tol)),
            check_every=int(check_every),
            lbfgs_history=int(lbfgs_history),
            lipschitz=str(lipschitz),
            compute_fitted=compute_fitted,
        )

        X = cp.asarray(X)
        fitted = cp.asarray(fitted) if fitted is not None else None
        return X, fitted

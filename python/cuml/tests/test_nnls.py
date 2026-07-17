#
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for cuml.solvers.nnls, with emphasis on the FISTA-style APG solver
that is built exclusively from raft / cuBLAS / cuSOLVER primitives.

Each shape and dtype combination is checked against ``scipy.optimize.nnls``
on three accuracy facets:

* the residual 2-norm should be no worse than scipy's (within a small slack),
* the KKT residual ``max_j |min(x_j, g_j)|`` should be small relative to the
  scale of ``A^T b``, and
* every coefficient should be non-negative.
"""

import cupy as cp
import numpy as np
import pytest
from scipy.optimize import nnls as scipy_nnls

from cuml.solvers import nnls as cuml_nnls


def _kkt_residual(A, x, b):
    """``max_j |min(x_j, g_j)|`` for the smooth NNLS objective."""
    g = A.T @ (A @ x - b)
    return float(np.max(np.abs(np.minimum(x, g))))


def _kkt_scale(A, b):
    return max(1.0, float(np.max(np.abs(A.T @ b))))


def _check_solution(A, x, b, *, dtype, residual_slack=1.5e-3, kkt_rel=1e-2):
    assert x.shape == (A.shape[1],)
    assert np.all(x >= -1e-6), "negative entries detected"
    x = np.maximum(x, 0.0)

    x_ref, rnorm_ref = scipy_nnls(A.astype(np.float64), b.astype(np.float64))
    rnorm = float(np.linalg.norm(A @ x - b))

    # APG should match scipy's residual to within float-precision slack,
    # scaled by ||b|| so the bound stays meaningful for under-determined
    # problems where rnorm_ref == 0 and the achievable residual is dominated
    # by float-roundoff times the problem scale.
    b_scale = max(1.0, float(np.linalg.norm(b)))
    abs_slack = (1e-3 if dtype == np.float32 else 1e-6) * b_scale
    assert rnorm <= rnorm_ref * (1.0 + residual_slack) + abs_slack, (
        f"residual {rnorm:.6g} > scipy residual {rnorm_ref:.6g} "
        f"(slack {residual_slack:.1%} + {abs_slack:.1e})"
    )

    # KKT residual should be small relative to the scale of A^T b.
    kkt = _kkt_residual(A.astype(np.float64), x.astype(np.float64),
                        b.astype(np.float64))
    scale = _kkt_scale(A.astype(np.float64), b.astype(np.float64))
    assert kkt <= kkt_rel * scale, (
        f"KKT residual {kkt:.3e} > {kkt_rel:.1e} * scale {scale:.3e}"
    )


def _make_tall(n_rows, n_cols, *, seed, dtype, sparsity=0.5):
    """Random tall problem with a known non-negative ground truth."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n_rows, n_cols)).astype(dtype)
    x_true = rng.uniform(0.0, 2.0, n_cols).astype(dtype)
    mask = rng.random(n_cols) < sparsity
    x_true[mask] = 0.0
    noise = (0.01 if dtype == np.float64 else 0.05) * rng.standard_normal(
        n_rows).astype(dtype)
    b = (A @ x_true + noise).astype(dtype)
    return A, b


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "n_rows,n_cols",
    [
        (200, 50),     # mildly tall
        (2000, 200),   # tall
        (200, 200),    # square
        (200, 500),    # wide / under-determined
    ],
)
def test_apg_random_dense(dtype, n_rows, n_cols):
    A, b = _make_tall(n_rows, n_cols, seed=0, dtype=dtype)

    x, rnorm = cuml_nnls(A, b, solver="apg", maxiter=2000,
                         tol=(1e-5 if dtype == np.float32 else 1e-7))
    x = cp.asnumpy(x)

    _check_solution(A, x, b, dtype=dtype)
    assert rnorm == pytest.approx(float(np.linalg.norm(A @ x - b)),
                                  rel=1e-3, abs=1e-4)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_apg_rank_deficient(dtype):
    """A duplicated column produces a rank-deficient Gram matrix; APG should
    still converge to a valid (possibly non-unique) NNLS solution."""
    rng = np.random.default_rng(7)
    n_rows, n_cols = 400, 30
    A = rng.standard_normal((n_rows, n_cols)).astype(dtype)
    # Force two columns to be linearly dependent.
    A[:, 5] = A[:, 4] * dtype(0.5) + dtype(1e-6) * rng.standard_normal(
        n_rows).astype(dtype)
    x_true = np.maximum(rng.standard_normal(n_cols), 0.0).astype(dtype)
    b = (A @ x_true + dtype(0.01) * rng.standard_normal(n_rows)).astype(dtype)

    x, _ = cuml_nnls(A, b, solver="apg", maxiter=4000,
                     tol=(1e-5 if dtype == np.float32 else 1e-7))
    x = cp.asnumpy(x)

    # Tolerate a slightly larger residual slack here because rank-deficient
    # problems have a continuum of optima with similar residuals.
    _check_solution(A, x, b, dtype=dtype, residual_slack=5e-3, kkt_rel=5e-2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_apg_msa_like(dtype):
    """An ``A`` shape that mimics a Mutation Signature Analysis NNLS step
    (tall, modest n_cols).  Many calls to ``nnls`` are made on the same A
    in that workload, but here we just verify a single fit."""
    n_rows, n_cols = 1536, 65
    rng = np.random.default_rng(42)
    # A with mostly small positive entries, like a signature matrix.
    A = np.abs(rng.standard_normal((n_rows, n_cols))).astype(dtype)
    x_true = rng.uniform(0.0, 5.0, n_cols).astype(dtype)
    x_true[rng.random(n_cols) < 0.4] = 0.0
    b = (A @ x_true + dtype(0.1) * rng.standard_normal(n_rows)).astype(dtype)

    x, _ = cuml_nnls(A, b, solver="apg", maxiter=3000,
                     tol=(1e-5 if dtype == np.float32 else 1e-7))
    x = cp.asnumpy(x)

    _check_solution(A, x, b, dtype=dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_apg_zero_input(dtype):
    """An all-zero ``A`` is degenerate; the solver should return x == 0."""
    A = np.zeros((10, 4), dtype=dtype)
    b = np.ones(10, dtype=dtype)

    x, rnorm = cuml_nnls(A, b, solver="apg", maxiter=100)
    x = cp.asnumpy(x)
    assert np.allclose(x, 0.0)
    assert rnorm == pytest.approx(float(np.linalg.norm(b)), rel=1e-5)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_apg_lipschitz_modes_agree(dtype):
    """Power-iteration and SVD-based Lipschitz estimates should yield the
    same final solution (up to optimisation tolerance)."""
    A, b = _make_tall(300, 40, seed=11, dtype=dtype)

    x_pow, _ = cuml_nnls(A, b, solver="apg", maxiter=4000,
                        tol=(1e-5 if dtype == np.float32 else 1e-8),
                        lipschitz="power")
    x_svd, _ = cuml_nnls(A, b, solver="apg", maxiter=4000,
                        tol=(1e-5 if dtype == np.float32 else 1e-8),
                        lipschitz="svd")

    x_pow = cp.asnumpy(x_pow)
    x_svd = cp.asnumpy(x_svd)
    rel = (1e-3 if dtype == np.float32 else 1e-6)
    abs_ = (1e-3 if dtype == np.float32 else 1e-6)
    np.testing.assert_allclose(x_pow, x_svd, rtol=rel, atol=abs_)


def test_apg_invalid_lipschitz():
    A = np.eye(3, dtype=np.float32)
    b = np.ones(3, dtype=np.float32)
    with pytest.raises(ValueError):
        cuml_nnls(A, b, solver="apg", lipschitz="not-a-method")


def test_apg_unknown_solver():
    A = np.eye(3, dtype=np.float32)
    b = np.ones(3, dtype=np.float32)
    with pytest.raises(ValueError):
        cuml_nnls(A, b, solver="unknown")


# ---------------------------------------------------------------------------
# Batched, masked, shared-A NNLS (cuml.solvers.nnls_batched)
# ---------------------------------------------------------------------------

from cuml.solvers import nnls_batched as cuml_nnls_batched

_BATCHED_SOLVERS = ["lawson", "lawson_multikernel", "apg", "cd", "sgd", "lbfgs"]


def _msa_reference(A, B, masks, b_index):
    """Per-problem scipy reference matching MSA run_NNLS.nnls_batched."""
    n_cols, n_problems = masks.shape
    out_weights = np.zeros((n_cols, n_problems), dtype=np.float64)
    out_fitted = np.zeros((A.shape[0], n_problems), dtype=np.float64)
    Ad = A.astype(np.float64)
    Bd = B.astype(np.float64)
    for j in range(n_problems):
        cols = np.flatnonzero(masks[:, j])
        if cols.size == 0:
            continue
        w, _ = scipy_nnls(Ad[:, cols], Bd[:, b_index[j]])
        out_weights[cols, j] = w
        out_fitted[:, j] = Ad[:, cols] @ w
    return out_weights, out_fitted


def _check_batched(A, B, masks, b_index, out_weights, out_fitted, *, dtype,
                   residual_slack=5e-3, kkt_rel=5e-2):
    n_cols, P = masks.shape
    assert out_weights.shape == (n_cols, P)
    assert np.all(out_weights >= -1e-5), "negative weights detected"

    w_ref, _ = _msa_reference(A, B, masks, b_index)
    Ad = A.astype(np.float64)
    Bd = B.astype(np.float64)
    b_scale_all = max(1.0, float(np.max(np.linalg.norm(Bd[:, b_index], axis=0))))
    abs_slack = (2e-3 if dtype == np.float32 else 1e-6) * b_scale_all

    for j in range(P):
        cols = np.flatnonzero(masks[:, j])
        off = np.ones(n_cols, dtype=bool)
        off[cols] = False
        assert np.allclose(out_weights[off, j], 0.0, atol=1e-5), (
            f"problem {j}: masked-out weights are nonzero"
        )
        b_j = Bd[:, b_index[j]]
        r = float(np.linalg.norm(Ad[:, cols] @ out_weights[cols, j].astype(np.float64)
                                 - b_j))
        r_ref = float(np.linalg.norm(Ad[:, cols] @ w_ref[cols, j] - b_j))
        assert r <= r_ref * (1.0 + residual_slack) + abs_slack, (
            f"problem {j}: residual {r:.6g} > scipy {r_ref:.6g}"
        )
        g = Ad[:, cols].T @ (Ad[:, cols] @ out_weights[cols, j].astype(np.float64)
                             - b_j)
        kkt = float(np.max(np.abs(np.minimum(out_weights[cols, j], g)))) if cols.size else 0.0
        scale = max(1.0, float(np.max(np.abs(Ad[:, cols].T @ b_j))))
        assert kkt <= kkt_rel * scale, (
            f"problem {j}: KKT {kkt:.3e} > {kkt_rel:.1e} * {scale:.3e}"
        )

    assert out_fitted.shape == (A.shape[0], P)


def _solver_kw(solver, dtype):
    """Generous iteration budgets so the iterative backends converge."""
    if solver in ("lawson", "lawson_multikernel"):
        return dict(maxiter=0, tol=(1e-4 if dtype == np.float32 else 1e-8))
    return dict(maxiter=5000, tol=(1e-5 if dtype == np.float32 else 1e-8))


def _run_batched(A, B, masks, **kw):
    """Call the device-native nnls_batched and mirror (X, fitted) to host."""
    X, fitted = cuml_nnls_batched(A, B, masks, **kw)
    X = cp.asnumpy(X)
    fitted = None if fitted is None else cp.asnumpy(fitted)
    return X, fitted


@pytest.mark.parametrize("solver", _BATCHED_SOLVERS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("P", [1, 4, 64])
def test_batched_random_dense(solver, dtype, P):
    rng = np.random.default_rng(P)
    m, n = 128, 24
    A = rng.standard_normal((m, n)).astype(dtype)
    x_true = rng.uniform(0.0, 2.0, (n, P)).astype(dtype)
    x_true[rng.random((n, P)) < 0.5] = 0.0
    B = (A @ x_true + dtype(0.01) * rng.standard_normal((m, P))).astype(dtype)
    masks = np.ones((n, P), dtype=bool)
    b_index = np.arange(P)

    out_weights, out_fitted = _run_batched(
        A, B, masks, b_index=b_index, solver=solver, **_solver_kw(solver, dtype))
    _check_batched(A, B, masks, b_index, out_weights, out_fitted, dtype=dtype)


@pytest.mark.parametrize("solver", _BATCHED_SOLVERS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_batched_masked(solver, dtype):
    rng = np.random.default_rng(123)
    m, n, P = 96, 20, 16
    A = rng.standard_normal((m, n)).astype(dtype)
    x_true = rng.uniform(0.0, 3.0, (n, P)).astype(dtype)
    B = (A @ x_true + dtype(0.05) * rng.standard_normal((m, P))).astype(dtype)

    masks = rng.random((n, P)) < 0.6
    for p in range(P):
        if masks[:, p].sum() < 3:
            masks[rng.choice(n, 3, replace=False), p] = True
    b_index = np.arange(P)

    out_weights, out_fitted = _run_batched(
        A, B, masks, b_index=b_index, solver=solver, **_solver_kw(solver, dtype))
    _check_batched(A, B, masks, b_index, out_weights, out_fitted, dtype=dtype)


@pytest.mark.parametrize("solver", _BATCHED_SOLVERS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_batched_msa_like(solver, dtype):
    """Shape mimicking a mutation-signature-attribution round."""
    rng = np.random.default_rng(7)
    m, n, P = 96, 65, 128
    A = np.abs(rng.standard_normal((m, n))).astype(dtype)
    x_true = rng.uniform(0.0, 5.0, (n, P)).astype(dtype)
    x_true[rng.random((n, P)) < 0.5] = 0.0
    B = (A @ x_true + dtype(0.1) * rng.standard_normal((m, P))).astype(dtype)
    masks = x_true > 0.0
    b_index = np.arange(P)

    out_weights, out_fitted = _run_batched(
        A, B, masks, b_index=b_index, solver=solver, **_solver_kw(solver, dtype))

    if solver in ("lawson", "lawson_multikernel", "cd"):
        _check_batched(A, B, masks, b_index, out_weights, out_fitted, dtype=dtype)
    else:
        _check_batched(A, B, masks, b_index, out_weights, out_fitted, dtype=dtype,
                       residual_slack=5e-2, kkt_rel=2e-1)


@pytest.mark.parametrize("solver", ["lawson"])
@pytest.mark.parametrize("dtype", [np.float64])
def test_batched_b_index_gather(solver, dtype):
    """Leave-one-out style: many problems share a few distinct B columns."""
    rng = np.random.default_rng(99)
    m, n = 96, 20
    n_targets = 8
    n_problems = 64
    A = np.abs(rng.standard_normal((m, n))).astype(dtype)
    B = np.abs(rng.standard_normal((m, n_targets))).astype(dtype)
    masks = rng.random((n, n_problems)) < 0.7
    for p in range(n_problems):
        if masks[:, p].sum() < 3:
            masks[rng.choice(n, 3, replace=False), p] = True
    b_index = rng.integers(0, n_targets, size=n_problems)

    out_weights, out_fitted = _run_batched(
        A, B, masks, b_index=b_index, solver=solver, **_solver_kw(solver, dtype))
    _check_batched(A, B, masks, b_index, out_weights, out_fitted, dtype=dtype)


def test_batched_fitted_matches_AX():
    rng = np.random.default_rng(0)
    m, n, P = 64, 12, 8
    A = np.abs(rng.standard_normal((m, n))).astype(np.float64)
    B = np.abs(rng.standard_normal((m, P))).astype(np.float64)
    masks = np.ones((n, P), dtype=bool)

    out_weights, out_fitted = _run_batched(A, B, masks, solver="lawson")
    np.testing.assert_allclose(out_fitted, A @ out_weights, rtol=1e-6, atol=1e-6)


def test_batched_returns_device_arrays():
    """nnls_batched is device-native: it returns cupy arrays, and fitted is
    None when compute_fitted=False (skipping the extra matmul)."""
    rng = np.random.default_rng(3)
    A = np.abs(rng.standard_normal((32, 6))).astype(np.float64)
    B = np.abs(rng.standard_normal((32, 5))).astype(np.float64)
    masks = np.ones((6, 5), dtype=bool)

    X, fitted = cuml_nnls_batched(A, B, masks, solver="lawson")
    assert isinstance(X, cp.ndarray)
    assert isinstance(fitted, cp.ndarray)
    assert X.shape == (6, 5)
    assert fitted.shape == (32, 5)

    X2, fitted2 = cuml_nnls_batched(
        A, B, masks, solver="lawson", compute_fitted=False)
    assert isinstance(X2, cp.ndarray)
    assert fitted2 is None
    np.testing.assert_allclose(cp.asnumpy(X2), cp.asnumpy(X), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("order", ["C", "F"])
def test_batched_mask_order_parity(order):
    """The result must be identical whether the mask is C- or F-contiguous.
    F-contiguous is the zero-copy device path; C-contiguous is copied to F
    inside the Cython layer. Both must produce the same solution."""
    rng = np.random.default_rng(order == "F")
    m, n, P = 96, 20, 32
    A = np.abs(rng.standard_normal((m, n))).astype(np.float64)
    x_true = rng.uniform(0.0, 3.0, (n, P)).astype(np.float64)
    x_true[rng.random((n, P)) < 0.5] = 0.0
    B = (A @ x_true).astype(np.float64)

    masks_bool = rng.random((n, P)) < 0.6
    for p in range(P):
        if masks_bool[:, p].sum() < 3:
            masks_bool[rng.choice(n, 3, replace=False), p] = True

    masks = np.asarray(masks_bool, order=order)
    assert masks.flags["%s_CONTIGUOUS" % order]
    b_index = np.arange(P)

    X, _ = cuml_nnls_batched(A, B, masks, b_index=b_index, solver="lawson",
                             **_solver_kw("lawson", np.float64))
    # Reference: the opposite order.
    other = "C" if order == "F" else "F"
    X_other, _ = cuml_nnls_batched(
        A, B, np.asarray(masks_bool, order=other), b_index=b_index,
        solver="lawson", **_solver_kw("lawson", np.float64))
    np.testing.assert_allclose(cp.asnumpy(X), cp.asnumpy(X_other),
                               rtol=1e-6, atol=1e-6)


def test_batched_on_device_l2_scoring_parity():
    """The GPU boundary (MSA batched_solve_and_score) reduces fitted to the
    L2_normalised_by_first similarity on the device. Verify that cupy reduction
    matches the numpy reference for the same fitted vectors."""
    rng = np.random.default_rng(2024)
    m, n = 96, 24
    n_targets, P = 10, 48
    A = np.abs(rng.standard_normal((m, n))).astype(np.float64)
    B = np.abs(rng.standard_normal((m, n_targets))).astype(np.float64)
    masks_bool = rng.random((n, P)) < 0.7
    for p in range(P):
        if masks_bool[:, p].sum() < 3:
            masks_bool[rng.choice(n, 3, replace=False), p] = True
    masks = np.asarray(masks_bool, order="F")
    b_index = rng.integers(0, n_targets, size=P)
    norm_obs = np.linalg.norm(B, axis=0)

    A_dev = cp.asarray(A)
    B_dev = cp.asfortranarray(cp.asarray(B))
    bi_dev = cp.asarray(b_index)
    _, fitted = cuml_nnls_batched(A_dev, B_dev, cp.asarray(masks),
                                  b_index=bi_dev, solver="lawson",
                                  **_solver_kw("lawson", np.float64))
    # On-device L2_normalised_by_first similarity.
    resid = cp.linalg.norm(B_dev[:, bi_dev] - fitted, axis=0)
    sims_dev = cp.asnumpy(1.0 - resid / cp.asarray(norm_obs)[bi_dev])

    # numpy reference from the same fitted vectors.
    fitted_np = cp.asnumpy(fitted)
    sims_ref = 1.0 - np.linalg.norm(B[:, b_index] - fitted_np, axis=0) / norm_obs[b_index]
    np.testing.assert_allclose(sims_dev, sims_ref, rtol=1e-6, atol=1e-6)


def test_batched_unknown_solver():
    A = np.eye(4, dtype=np.float32)
    B = np.ones((4, 2), dtype=np.float32)
    masks = np.ones((4, 2), dtype=bool)
    with pytest.raises(ValueError):
        cuml_nnls_batched(A, B, masks, solver="nope")


def test_batched_bad_mask_shape():
    A = np.eye(4, dtype=np.float32)
    B = np.ones((4, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        cuml_nnls_batched(A, B, np.ones((3, 4), dtype=bool))

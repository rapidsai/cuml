/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/common/export.hpp>

#include <cstdint>

namespace raft {
class handle_t;
}

namespace CUML_EXPORT ML {
namespace Solver {

/**
 * @brief Method for estimating the gradient Lipschitz constant L of the smooth
 * NNLS objective f(x) = 1/2 ||A x - b||^2.
 *
 * For NNLS we have ∇f(x) = A^T (A x - b), so L = ||A||_2^2 = sigma_max(A)^2.
 */
enum class NnlsLipschitzMethod {
  POWER_ITERATION = 0,  ///< Cheap iterative estimate via two gemv per step.
  SVD             = 1,  ///< Exact value via cuSOLVER SVD on a copy of A.
  USER_SUPPLIED   = 2   ///< Use NnlsApgParams::lipschitz_value as-is.
};

/**
 * Parameters for the FISTA / accelerated projected gradient NNLS solver.
 */
struct NnlsApgParams {
  int    max_iter         = 1000;
  double tol              = 1e-6;
  int    check_every      = 10;
  bool   restart          = true;
  NnlsLipschitzMethod lipschitz_method = NnlsLipschitzMethod::POWER_ITERATION;
  double lipschitz_value  = 0.0;  ///< Used only when lipschitz_method == USER_SUPPLIED.
  int    power_iter       = 30;   ///< Power-iteration steps for L estimation.
  double lipschitz_safety = 1.05; ///< Multiplicative safety factor on L.
};

/**
 * Solve the Non-Negative Least Squares problem
 *
 *   argmin_x  1/2 || A x - b ||_2^2,  subject to x >= 0
 *
 * by viewing it as a convex quadratic program and applying FISTA-style
 * accelerated projected gradient (APG) iterations.  The Hessian G = A^T A is
 * never formed explicitly; each iteration performs two gemv calls (A y and
 * A^T r), one fused projection elementwise op, optional adaptive restart,
 * and a Nesterov momentum combine.  All work is performed via the standard
 * raft / cuBLAS / cuSOLVER primitives (raft::linalg::gemv, ::axpy, ::dot,
 * ::map, ::map_reduce, ::svdJacobi); no custom CUDA kernels are used.
 *
 * @param handle  raft handle.  All work runs on its main stream.
 * @param A       column-major coefficient matrix of shape (n_rows, n_cols).
 * @param n_rows  number of rows of A (length of b).
 * @param n_cols  number of columns of A (length of x).
 * @param b       right-hand-side vector of length n_rows.
 * @param x       output solution vector of length n_cols.  Pre-existing
 *                contents are overwritten.
 * @param params  solver parameters (see NnlsApgParams).
 * @return        number of outer iterations actually performed.
 */
int nnlsApg(raft::handle_t& handle,
            const float* A,
            int n_rows,
            int n_cols,
            const float* b,
            float* x,
            const NnlsApgParams& params);

int nnlsApg(raft::handle_t& handle,
            const double* A,
            int n_rows,
            int n_cols,
            const double* b,
            double* x,
            const NnlsApgParams& params);

/**
 * @brief Solver backend selector for the batched NNLS entry point.
 *
 * All five backends service the same batched, masked, shared-A contract
 * (see nnlsBatched).  LAWSON is an exact active-set method run as one CUDA
 * block per problem; the remaining four minimise the same convex QP with a
 * projected iterative scheme (also one block per problem) and are provided
 * for completeness / cross-checking.
 */
enum class NnlsBatchedSolver {
  LAWSON = 0,  ///< Lawson-Hanson active-set (exact, best for small n_cols).
  APG    = 1,  ///< FISTA-style accelerated projected gradient.
  CD     = 2,  ///< Coordinate descent on the QP.
  SGD    = 3,  ///< Projected gradient descent (ISTA).
  LBFGS  = 4,  ///< Projected limited-memory BFGS.
  LAWSON_MULTIKERNEL = 5  ///< Lawson-Hanson advanced one step over the whole
                          ///< batch per global kernel launch; factor maintained
                          ///< in global memory with batched cuBLAS/cuSOLVER
                          ///< primitives, stopping polled from pinned memory.
};

/**
 * Parameters shared by every batched NNLS backend.
 */
struct NnlsBatchedParams {
  NnlsBatchedSolver solver = NnlsBatchedSolver::LAWSON;
  int    max_iter     = 0;      ///< 0 => per-solver default.
  double tol          = 1e-6;   ///< Relative KKT tolerance.
  int    check_every  = 10;     ///< Iterations between convergence checks.
  int    lbfgs_history = 5;     ///< History length for the LBFGS backend.

  // Lipschitz-constant estimation for the gradient backends (APG, SGD).
  // L = sigma_max(A)^2 is estimated once on the shared A and reused as the
  // step size 1/L for every problem (submatrices only shrink the spectrum,
  // so a single global L is a valid step for the whole masked batch).
  NnlsLipschitzMethod lipschitz_method = NnlsLipschitzMethod::POWER_ITERATION;
  double lipschitz_value  = 0.0;   ///< Used only for USER_SUPPLIED.
  int    power_iter       = 30;    ///< Power-iteration steps for L.
  double lipschitz_safety = 1.05;  ///< Multiplicative safety factor on L.
};

/**
 * Solve a batch of Non-Negative Least Squares problems that share the same
 * coefficient matrix but differ by right-hand side and active-column support:
 *
 *   for p in [0, n_problems):
 *     X[:, p] = argmin_{x >= 0, x[j]=0 for masks[j,p]==0}
 *                 1/2 || A[:, support_p] x[support_p] - B[:, p] ||_2^2
 *
 * The shared matrix A stays resident; its Gram matrix G = A^T A and the RHS
 * projections C = A^T B are formed once (via cuBLAS) and reused by every
 * problem.  masks selects the active support per problem; masked-out
 * coordinates of X are pinned to zero.
 *
 * @param handle       raft handle (all work on its main stream).
 * @param A            column-major coefficient matrix, shape (m, n).
 * @param m            number of rows of A (length of each B column).
 * @param n            number of columns of A (length of each X column).
 * @param B            column-major RHS matrix, shape (m, n_problems).
 * @param n_problems   number of problems / columns of B and X.
 * @param masks        column-major uint8 matrix, shape (n_signatures, n_problems)
 *                     i.e. (n, n_problems), F-contiguous; element (j, p) lives at
 *                     masks[p*n + j] and is nonzero iff column j is active for
 *                     problem p.  May be null, meaning every column is active for
 *                     every problem.  (The byte layout is identical to a
 *                     row-major (n_problems, n) array; only the interpretation
 *                     differs, so the per-block access masks[p*n + j] is
 *                     unchanged and coalesced over the signature index j.)
 * @param X            output solutions, column-major (n, n_problems).  Masked-out
 *                     rows are written as 0.
 * @param fitted       optional output A @ X, column-major (m, n_problems).  May
 *                     be null to skip the final gemm.
 * @param params       solver selection and per-backend knobs.
 */
void nnlsBatched(raft::handle_t& handle,
                 const float* A,
                 int m,
                 int n,
                 const float* B,
                 int n_problems,
                 const std::uint8_t* masks,
                 float* X,
                 float* fitted,
                 const NnlsBatchedParams& params);

void nnlsBatched(raft::handle_t& handle,
                 const double* A,
                 int m,
                 int n,
                 const double* B,
                 int n_problems,
                 const std::uint8_t* masks,
                 double* X,
                 double* fitted,
                 const NnlsBatchedParams& params);

}  // namespace Solver
}  // end namespace CUML_EXPORT ML

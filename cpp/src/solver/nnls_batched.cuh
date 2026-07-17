/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "nnls_apg.cuh"      // detail::power_iter_lipschitz / svd_lipschitz
#include "nnls_lawson.cuh"   // detail::nnls_lawson_batched_kernel + smem helpers
#include "nnls_lawson_multikernel.cuh"  // detail::nnls_lawson_multikernel_dispatch
#include "nnls_qp_batched.cuh"

#include <cuml/common/utils.hpp>
#include <cuml/solvers/nnls.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/linalg/gemm.cuh>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <optional>

namespace ML {
namespace Solver {
namespace detail {

// nnls_set_smem_attr (shared dynamic-shared-memory opt-in helper) lives in
// nnls_lawson.cuh and is reused by every backend launcher below.

/** Estimate L = sigma_max(A)^2 (shared step for the gradient backends). */
template <typename T>
T nnls_batched_lipschitz(raft::handle_t& handle,
                         const T*        A,
                         int             m,
                         int             n,
                         const NnlsBatchedParams& params)
{
  cudaStream_t stream = handle.get_stream();
  switch (params.lipschitz_method) {
    case NnlsLipschitzMethod::USER_SUPPLIED: return static_cast<T>(params.lipschitz_value);
    case NnlsLipschitzMethod::SVD:
      return svd_lipschitz<T>(handle, A, m, n) * static_cast<T>(params.lipschitz_safety);
    case NnlsLipschitzMethod::POWER_ITERATION:
    default: {
      rmm::device_uvector<T> v_pi(n, stream);
      rmm::device_uvector<T> u_pi(m, stream);
      rmm::device_scalar<T>  d_scratch(stream);
      T L = power_iter_lipschitz<T>(handle, A, m, n, params.power_iter, v_pi.data(), u_pi.data(),
                                    d_scratch);
      return L * static_cast<T>(params.lipschitz_safety);
    }
  }
}

template <typename T>
void nnls_batched_impl(raft::handle_t&          handle,
                       const T*                 A,
                       int                      m,
                       int                      n,
                       const T*                 B,
                       int                      P,
                       const std::uint8_t*      masks,
                       T*                        X,
                       T*                        fitted,
                       const NnlsBatchedParams& params)
{
  raft::common::nvtx::range fun_scope(
    "ML::Solver::nnlsBatched(%d, %d, %d)", m, n, P);
  ASSERT(m >= 1, "ML::Solver::nnlsBatched: m must be >= 1.");
  ASSERT(n >= 1, "ML::Solver::nnlsBatched: n must be >= 1.");
  ASSERT(P >= 1, "ML::Solver::nnlsBatched: n_problems must be >= 1.");

  cudaStream_t stream = handle.get_stream();

  // ---- Precompute the resident Gram / RHS projections ---------------------
  // G = A^T A  (n x n),  C = A^T B  (n x P).  Formed once and reused by every
  // problem in the batch.  A col-major (m, n) buffer viewed as row-major
  // (n, m) is exactly A^T, so raft's mdspan gemm expresses the transpose
  // through the operand layout rather than a flag.
  auto G = raft::make_device_matrix<T, int, raft::col_major>(handle, n, n);
  auto C = raft::make_device_matrix<T, int, raft::col_major>(handle, n, P);

  // gemm's mdspan overload shares one ElementType across all operands, so the
  // read-only inputs are wrapped in non-const views (gemm never writes them).
  auto* A_mut  = const_cast<T*>(A);
  auto  At_view = raft::make_device_matrix_view<T, int, raft::row_major>(A_mut, n, m);  // A^T (n x m)
  auto  A_view  = raft::make_device_matrix_view<T, int, raft::col_major>(A_mut, m, n);  // A   (m x n)
  auto  B_view  = raft::make_device_matrix_view<T, int, raft::col_major>(const_cast<T*>(B), m, P);
  raft::linalg::gemm(handle, At_view, A_view, G.view());
  raft::linalg::gemm(handle, At_view, B_view, C.view());
  const T one  = T(1);
  const T zero = T(0);

  // ---- Dispatch to the selected backend -----------------------------------
  int max_iter = params.max_iter;
  const T tol  = static_cast<T>(params.tol);

  if (params.solver == NnlsBatchedSolver::LAWSON) {
    if (max_iter <= 0) max_iter = 3 * n + 1;
    // The kernel is templated on the block size; the dispatcher selects an
    // instantiation from the kernel occupancy and applies the smem carveout.
    auto G_view = raft::make_const_mdspan(G.view());
    auto C_view = raft::make_const_mdspan(C.view());
    auto X_view = raft::make_device_matrix_view<T, int, raft::col_major>(X, n, P);
    std::optional<raft::device_matrix_view<const std::uint8_t, int, raft::col_major>> M_view;
    if (masks != nullptr)
      M_view = raft::make_device_matrix_view<const std::uint8_t, int, raft::col_major>(masks, n, P);
    nnls_lawson_batched_dispatch<T>(handle, G_view, C_view, M_view, X_view, max_iter, tol);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  } else if (params.solver == NnlsBatchedSolver::LAWSON_MULTIKERNEL) {
    if (max_iter <= 0) max_iter = 3 * n + 1;
    auto G_view = raft::make_const_mdspan(G.view());
    auto C_view = raft::make_const_mdspan(C.view());
    auto X_view = raft::make_device_matrix_view<T, int, raft::col_major>(X, n, P);
    std::optional<raft::device_matrix_view<const std::uint8_t, int, raft::col_major>> M_view;
    if (masks != nullptr)
      M_view = raft::make_device_matrix_view<const std::uint8_t, int, raft::col_major>(masks, n, P);
    nnls_lawson_multikernel_dispatch<T>(handle, G_view, C_view, M_view, X_view, max_iter, tol);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  } else {
    // Gradient / coordinate backends operate on the QP in Gram form.
    const int check_every = params.check_every >= 1 ? params.check_every : 10;

    // Step size 1/L for the gradient methods.  A single global L is valid for
    // every masked sub-problem (submatrices only shrink the spectrum).
    T inv_L = T(0);
    if (params.solver == NnlsBatchedSolver::APG ||
        params.solver == NnlsBatchedSolver::SGD ||
        params.solver == NnlsBatchedSolver::LBFGS) {
      T L = nnls_batched_lipschitz<T>(handle, A, m, n, params);
      if (!(L > T(0))) {
        // A == 0: every x >= 0 is optimal; return zeros.
        thrust::fill_n(thrust::cuda::par.on(stream), X, static_cast<std::size_t>(n) * P, T(0));
        if (fitted != nullptr)
          thrust::fill_n(thrust::cuda::par.on(stream), fitted, static_cast<std::size_t>(m) * P,
                         T(0));
        handle.sync_stream(stream);
        return;
      }
      inv_L = T(1) / L;
    }

    switch (params.solver) {
      case NnlsBatchedSolver::SGD: {
        if (max_iter <= 0) max_iter = 1000;
        const std::size_t smem = qp_grad_smem_bytes<T>(n);
        ASSERT(nnls_set_smem_attr(handle, nnls_sgd_batched_kernel<T>, smem),
               "ML::Solver::nnlsBatched: required shared memory (%zu B) exceeds the device "
               "per-block opt-in limit; reduce n_cols or pick a different solver.",
               smem);
        nnls_sgd_batched_kernel<T><<<P, QP_BLOCK_SIZE, smem, stream>>>(
          G.data_handle(), C.data_handle(), n, masks, X, inv_L, max_iter, check_every, tol);
        break;
      }
      case NnlsBatchedSolver::APG: {
        if (max_iter <= 0) max_iter = 1000;
        const std::size_t smem = qp_grad_smem_bytes<T>(n);
        ASSERT(nnls_set_smem_attr(handle, nnls_apg_batched_kernel<T>, smem),
               "ML::Solver::nnlsBatched: required shared memory (%zu B) exceeds the device "
               "per-block opt-in limit; reduce n_cols or pick a different solver.",
               smem);
        nnls_apg_batched_kernel<T><<<P, QP_BLOCK_SIZE, smem, stream>>>(
          G.data_handle(), C.data_handle(), n, masks, X, inv_L, max_iter, check_every, tol);
        break;
      }
      case NnlsBatchedSolver::CD: {
        if (max_iter <= 0) max_iter = 1000;
        const std::size_t smem = qp_grad_smem_bytes<T>(n);
        ASSERT(nnls_set_smem_attr(handle, nnls_cd_batched_kernel<T>, smem),
               "ML::Solver::nnlsBatched: required shared memory (%zu B) exceeds the device "
               "per-block opt-in limit; reduce n_cols or pick a different solver.",
               smem);
        nnls_cd_batched_kernel<T><<<P, QP_BLOCK_SIZE, smem, stream>>>(
          G.data_handle(), C.data_handle(), n, masks, X, max_iter, check_every, tol);
        break;
      }
      case NnlsBatchedSolver::LBFGS: {
        if (max_iter <= 0) max_iter = 500;
        int hist = params.lbfgs_history;
        if (hist < 1) hist = 1;
        if (hist > QP_LBFGS_MAX_HISTORY) hist = QP_LBFGS_MAX_HISTORY;
        const std::size_t smem = qp_lbfgs_smem_bytes<T>(n, hist);
        ASSERT(nnls_set_smem_attr(handle, nnls_lbfgs_batched_kernel<T>, smem),
               "ML::Solver::nnlsBatched: required shared memory (%zu B) exceeds the device "
               "per-block opt-in limit; reduce n_cols or pick a different solver.",
               smem);
        nnls_lbfgs_batched_kernel<T><<<P, QP_BLOCK_SIZE, smem, stream>>>(
          G.data_handle(), C.data_handle(), n, masks, X, inv_L, max_iter, hist, check_every, tol);
        break;
      }
      default: THROW("ML::Solver::nnlsBatched: unknown solver.");
    }
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  // ---- Optional fitted = A @ X  (m x P) -----------------------------------
  if (fitted != nullptr) {
    raft::linalg::gemm(handle, /*trans_a=*/false, /*trans_b=*/false, m, P, n, &one, A, m, X, n,
                       &zero, fitted, m, stream);
  }
}

}  // namespace detail
}  // namespace Solver
}  // namespace ML

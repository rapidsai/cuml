/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/common/utils.hpp>
#include <cuml/solvers/nnls.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/nvtx.hpp>
#include <raft/core/operators.hpp>
#include <raft/linalg/axpy.cuh>
#include <raft/linalg/dot.cuh>
#include <raft/linalg/gemv.cuh>
#include <raft/linalg/map.cuh>
#include <raft/linalg/map_reduce.cuh>
#include <raft/linalg/svd.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace ML {
namespace Solver {
namespace detail {

/**
 * @brief Estimate the largest singular value of A (and hence L = ||A||_2^2)
 * with a few power iterations on A^T A.
 *
 * Uses only raft::linalg::gemv (cuBLAS), raft::linalg::dot (cuBLAS) and
 * raft::linalg::map (a single fused CUDA kernel for the v <- v / ||v||
 * normalization).  No allocations beyond the two scratch vectors `u` and
 * `v` provided by the caller.
 *
 * On entry `v` is overwritten with an initial guess; on exit it holds the
 * approximate top right singular vector.  The estimated L = sigma_max(A)^2
 * is returned on the host.
 */
template <typename T>
T power_iter_lipschitz(raft::handle_t&        handle,
                       const T*               A,
                       int                    m,
                       int                    n,
                       int                    n_iter,
                       T*                     v,
                       T*                     u,
                       rmm::device_scalar<T>& d_scratch)
{
  cudaStream_t stream = handle.get_stream();

  // Initialize v as the uniform unit vector 1 / sqrt(n).  Using a deterministic
  // start keeps the result reproducible across runs and is sufficient for
  // sigma_max so long as v has a non-zero component along the dominant
  // singular direction (true for any A that has a non-zero column).
  const T v0 = T(1) / std::sqrt(static_cast<T>(n));
  thrust::fill_n(thrust::cuda::par.on(stream), v, n, v0);

  T lambda = T(1);
  for (int k = 0; k < n_iter; ++k) {
    // u = A v
    raft::linalg::gemv(handle, A, m, n, v, u, /*trans_a=*/false, T(1), T(0), stream);

    // lambda = u . u  (= v^T A^T A v with v unit  ==> Rayleigh quotient on A^T A)
    raft::linalg::dot(handle,
                      raft::make_device_vector_view<const T, int>(u, m),
                      raft::make_device_vector_view<const T, int>(u, m),
                      raft::make_device_scalar_view<T>(d_scratch.data()));
    raft::update_host(&lambda, d_scratch.data(), 1, stream);

    // v = A^T u  (= A^T A v_prev)
    raft::linalg::gemv(handle, A, m, n, u, v, /*trans_a=*/true, T(1), T(0), stream);

    // ||v||
    raft::linalg::dot(handle,
                      raft::make_device_vector_view<const T, int>(v, n),
                      raft::make_device_vector_view<const T, int>(v, n),
                      raft::make_device_scalar_view<T>(d_scratch.data()));
    T vnorm_sq;
    raft::update_host(&vnorm_sq, d_scratch.data(), 1, stream);
    handle.sync_stream(stream);

    if (!(vnorm_sq > T(0))) break;
    const T inv_vnorm = T(1) / std::sqrt(vnorm_sq);
    raft::linalg::map(handle,
                      raft::make_device_vector_view<const T, int>(v, n),
                      raft::make_device_vector_view<T, int>(v, n),
                      [inv_vnorm] __device__(T x) { return x * inv_vnorm; });
  }
  return lambda;
}

/**
 * @brief Compute L = sigma_max(A)^2 exactly via cuSOLVER's Jacobi SVD.
 *
 * Allocates an m*n scratch copy of A because svdJacobi factors in place,
 * plus thin-SVD U (m*k) and V (n*k) workspaces because raft::linalg::svdJacobi
 * always invokes cuSOLVER with CUSOLVER_EIG_MODE_VECTOR and econ=1 and writes
 * into both buffers regardless of the gen_*_vec flags.  This is the costliest
 * Lipschitz mode and is intended for small matrices or for accuracy debugging;
 * the default mode is power iteration.
 */
template <typename T>
T svd_lipschitz(raft::handle_t& handle, const T* A, int m, int n)
{
  cudaStream_t     stream = handle.get_stream();
  const std::size_t A_size = static_cast<std::size_t>(m) * n;
  rmm::device_uvector<T> A_copy(A_size, stream);
  raft::copy(A_copy.data(), A, A_size, stream);

  const int              k = std::min(m, n);
  rmm::device_uvector<T> sing_vals(k, stream);
  rmm::device_uvector<T> U(static_cast<std::size_t>(m) * k, stream);
  rmm::device_uvector<T> V(static_cast<std::size_t>(n) * k, stream);

  raft::linalg::svdJacobi<T>(handle,
                             A_copy.data(),
                             m,
                             n,
                             sing_vals.data(),
                             U.data(),
                             V.data(),
                             /*gen_left_vec=*/false,
                             /*gen_right_vec=*/false,
                             /*tol=*/static_cast<T>(sizeof(T) == 4 ? 1e-7 : 1e-14),
                             /*max_sweeps=*/100,
                             stream);

  // cusolverDngesvdj returns singular values in descending order, so
  // sing_vals[0] is sigma_max.
  T sigma_max;
  raft::update_host(&sigma_max, sing_vals.data(), 1, stream);
  handle.sync_stream(stream);
  return sigma_max * sigma_max;
}

/**
 * @brief Core APG iteration for NNLS.  See cuml/solvers/nnls.hpp for the
 * problem statement and parameter semantics.
 */
template <typename T>
int nnls_apg_impl(raft::handle_t&     handle,
                  const T*            A,
                  int                 m,
                  int                 n,
                  const T*            b,
                  T*                  x_out,
                  const NnlsApgParams& params)
{
  raft::common::nvtx::range fun_scope("ML::Solver::nnlsApg(%d, %d)", m, n);
  ASSERT(m >= 1, "ML::Solver::nnlsApg: n_rows must be >= 1.");
  ASSERT(n >= 1, "ML::Solver::nnlsApg: n_cols must be >= 1.");
  ASSERT(params.max_iter >= 1, "ML::Solver::nnlsApg: max_iter must be >= 1.");
  ASSERT(params.check_every >= 1, "ML::Solver::nnlsApg: check_every must be >= 1.");

  cudaStream_t stream = handle.get_stream();

  // Working buffers.
  rmm::device_uvector<T> y(n, stream);
  rmm::device_uvector<T> x_old(n, stream);
  rmm::device_uvector<T> x_new(n, stream);
  rmm::device_uvector<T> r(m, stream);
  rmm::device_uvector<T> g(n, stream);
  rmm::device_uvector<T> v_pi(n, stream);  // power-iteration scratch
  rmm::device_uvector<T> u_pi(m, stream);  // power-iteration scratch
  rmm::device_scalar<T>  d_scratch(stream);

  // ---- Lipschitz constant L = ||A||_2^2 ------------------------------------
  T L;
  switch (params.lipschitz_method) {
    case NnlsLipschitzMethod::POWER_ITERATION:
      L = power_iter_lipschitz<T>(handle, A, m, n, params.power_iter, v_pi.data(), u_pi.data(), d_scratch);
      L *= static_cast<T>(params.lipschitz_safety);
      break;
    case NnlsLipschitzMethod::SVD:
      L = svd_lipschitz<T>(handle, A, m, n);
      L *= static_cast<T>(params.lipschitz_safety);
      break;
    case NnlsLipschitzMethod::USER_SUPPLIED:
      L = static_cast<T>(params.lipschitz_value);
      ASSERT(L > T(0),
             "ML::Solver::nnlsApg: USER_SUPPLIED Lipschitz mode requires "
             "lipschitz_value > 0 (got %g).",
             static_cast<double>(L));
      break;
    default: THROW("ML::Solver::nnlsApg: unknown Lipschitz method.");
  }
  // Degenerate case: A == 0  ==>  any x >= 0 is optimal; return zero.
  if (!(L > T(0))) {
    thrust::fill_n(thrust::cuda::par.on(stream), x_out, n, T(0));
    handle.sync_stream(stream);
    return 0;
  }
  const T inv_L = T(1) / L;

  // ---- Initial state -------------------------------------------------------
  thrust::fill_n(thrust::cuda::par.on(stream), y.data(), n, T(0));
  thrust::fill_n(thrust::cuda::par.on(stream), x_old.data(), n, T(0));
  thrust::fill_n(thrust::cuda::par.on(stream), x_new.data(), n, T(0));

  // Tolerance scale: max(1, ||A^T b||_inf).  Compute once.
  raft::linalg::gemv(handle, A, m, n, b, g.data(), /*trans_a=*/true, T(1), T(0), stream);
  raft::linalg::mapReduce<T, raft::abs_op, raft::max_op, std::uint32_t>(
    d_scratch.data(),
    static_cast<std::size_t>(n),
    T(0),
    raft::abs_op{},
    raft::max_op{},
    stream,
    g.data());
  T c_norm_inf;
  raft::update_host(&c_norm_inf, d_scratch.data(), 1, stream);
  handle.sync_stream(stream);

  const T scale = std::max(T(1), c_norm_inf);
  const T tol   = static_cast<T>(params.tol) * scale;

  // ---- APG main loop -------------------------------------------------------
  T   t_curr = T(1);
  int n_iter = 0;
  for (int k = 0; k < params.max_iter; ++k) {
    n_iter = k + 1;

    // 1) r = A y - b   (compute via two-stage:  r := b ; r := A y - r)
    raft::copy(r.data(), b, m, stream);
    raft::linalg::gemv(handle, A, m, n, y.data(), r.data(),
                       /*trans_a=*/false, T(1), T(-1), stream);

    // 2) g = A^T r
    raft::linalg::gemv(handle, A, m, n, r.data(), g.data(),
                       /*trans_a=*/true, T(1), T(0), stream);

    // 3) Projection: x_new = max(0, y - g / L)  (single fused map kernel)
    raft::linalg::map(
      handle,
      raft::make_device_vector_view<const T, int>(y.data(), n),
      raft::make_device_vector_view<const T, int>(g.data(), n),
      raft::make_device_vector_view<T, int>(x_new.data(), n),
      [inv_L] __device__(T yi, T gi) {
        T v = yi - gi * inv_L;
        return v > T(0) ? v : T(0);
      });

    // 4) KKT check every params.check_every iterations.  Residual is
    //    ||min(x_new, g)||_inf in absolute value.  At an optimum,
    //    min(x_j, g_j) == 0 for every j (KKT for nonnegativity).
    if ((k % params.check_every) == 0) {
      raft::linalg::mapReduce<T>(
        d_scratch.data(),
        static_cast<std::size_t>(n),
        T(0),
        [] __device__(T xi, T gi) {
          T m_ = xi < gi ? xi : gi;
          return m_ < T(0) ? -m_ : m_;
        },
        raft::max_op{},
        stream,
        x_new.data(),
        g.data());
      T kkt;
      raft::update_host(&kkt, d_scratch.data(), 1, stream);
      handle.sync_stream(stream);
      if (kkt < tol) {
        raft::copy(x_out, x_new.data(), n, stream);
        handle.sync_stream(stream);
        return n_iter;
      }
    }

    // 5) Adaptive (gradient) restart: if g . (x_new - x_old) > 0 the
    //    momentum direction is locally ascending; reset the Nesterov state.
    bool restart_now = false;
    if (params.restart && k > 0) {
      raft::linalg::mapReduce<T>(
        d_scratch.data(),
        static_cast<std::size_t>(n),
        T(0),
        [] __device__(T gi, T xn, T xo) { return gi * (xn - xo); },
        raft::add_op{},
        stream,
        g.data(),
        x_new.data(),
        x_old.data());
      T gd;
      raft::update_host(&gd, d_scratch.data(), 1, stream);
      handle.sync_stream(stream);
      if (gd > T(0)) restart_now = true;
    }

    // 6) Nesterov momentum combine: y = x_new + beta * (x_new - x_old)
    T t_next;
    T beta;
    if (restart_now) {
      t_next = T(1);
      beta   = T(0);
    } else {
      t_next = (T(1) + std::sqrt(T(1) + T(4) * t_curr * t_curr)) / T(2);
      beta   = (t_curr - T(1)) / t_next;
    }

    raft::linalg::map(
      handle,
      raft::make_device_vector_view<const T, int>(x_new.data(), n),
      raft::make_device_vector_view<const T, int>(x_old.data(), n),
      raft::make_device_vector_view<T, int>(y.data(), n),
      [beta] __device__(T xn, T xo) { return xn + beta * (xn - xo); });

    // 7) Shift state: x_old <- x_new, t_curr <- t_next
    raft::copy(x_old.data(), x_new.data(), n, stream);
    t_curr = t_next;
  }

  // Maxiter reached.
  raft::copy(x_out, x_new.data(), n, stream);
  handle.sync_stream(stream);
  return n_iter;
}

}  // namespace detail
}  // namespace Solver
}  // namespace ML

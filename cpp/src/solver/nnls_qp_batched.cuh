/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace ML {
namespace Solver {
namespace detail {

// One CUDA block per problem.  Each block minimises the convex QP
//
//   f(x) = 1/2 x^T G x - c^T x,   x >= 0,   x[j] = 0 where mask[j] == 0
//
// which is exactly the NNLS objective in Gram form (G = A^T A, c = A^T b).
// The four backends below (SGD, APG, CD, LBFGS) differ only in the step rule;
// they all read the resident, shared G and their own column of C, keep the
// per-problem working set in shared memory, and write x into column p of X.
//
// masks is column-major (n_signatures, n_problems) = (n, n_problems),
// F-contiguous: element (j, p) lives at masks[p*n + j].  Block p therefore
// reads its support at base `masks + p*n` and indexes `[j]` over signatures,
// which is contiguous (coalesced).  This byte layout is identical to a
// row-major (n_problems, n) array, so the access below is unchanged.

constexpr int QP_BLOCK_SIZE = 128;
constexpr int QP_WARP_SIZE  = 32;
constexpr int QP_N_WARPS    = QP_BLOCK_SIZE / QP_WARP_SIZE;
constexpr int QP_LBFGS_MAX_HISTORY = 8;

/** Block-wide sum of a per-thread value; result broadcast to every thread. */
template <typename T>
__device__ inline T qp_block_sum(T v, T* scratch)
{
  const int lane    = threadIdx.x & (QP_WARP_SIZE - 1);
  const int warp_id = threadIdx.x / QP_WARP_SIZE;
  for (int off = QP_WARP_SIZE / 2; off > 0; off >>= 1)
    v += __shfl_xor_sync(0xffffffff, v, off);
  if (lane == 0) scratch[warp_id] = v;
  __syncthreads();
  T total = T(0);
#pragma unroll
  for (int w = 0; w < QP_N_WARPS; ++w)
    total += scratch[w];
  __syncthreads();
  return total;
}

/** Block-wide max of a per-thread value; result broadcast to every thread. */
template <typename T>
__device__ inline T qp_block_max(T v, T* scratch)
{
  const int lane    = threadIdx.x & (QP_WARP_SIZE - 1);
  const int warp_id = threadIdx.x / QP_WARP_SIZE;
  for (int off = QP_WARP_SIZE / 2; off > 0; off >>= 1) {
    T o = __shfl_xor_sync(0xffffffff, v, off);
    if (o > v) v = o;
  }
  if (lane == 0) scratch[warp_id] = v;
  __syncthreads();
  T best = -std::numeric_limits<T>::infinity();
#pragma unroll
  for (int w = 0; w < QP_N_WARPS; ++w)
    if (scratch[w] > best) best = scratch[w];
  __syncthreads();
  return best;
}

/**
 * Gradient g = G x - c for the current x (cooperative over columns of G).
 * Masked coordinates keep x == 0 so they never contribute to the product.
 */
template <typename T>
__device__ inline void qp_gradient(const T* G, const T* c, const T* x, T* g, int n)
{
  for (int j = threadIdx.x; j < n; j += QP_BLOCK_SIZE) {
    T acc = -c[j];
    for (int k = 0; k < n; ++k)
      acc += G[j + k * n] * x[k];
    g[j] = acc;
  }
  __syncthreads();
}

/**
 * KKT residual  max_j |min(x_j, g_j)|  over active coordinates.  A masked-out
 * coordinate is pinned to 0 and excluded from the check.  Result is broadcast.
 */
template <typename T>
__device__ inline T qp_kkt_residual(
  const T* x, const T* g, const std::uint8_t* mask, int n, T* scratch)
{
  T v = T(0);
  for (int j = threadIdx.x; j < n; j += QP_BLOCK_SIZE) {
    if (mask == nullptr || mask[j] != 0) {
      T m_ = x[j] < g[j] ? x[j] : g[j];
      T a  = m_ < T(0) ? -m_ : m_;
      if (a > v) v = a;
    }
  }
  return qp_block_max<T>(v, scratch);
}

/** max_j |c_j| used to scale the relative KKT tolerance. */
template <typename T>
__device__ inline T qp_scale(const T* c, int n, T* scratch)
{
  T v = T(0);
  for (int j = threadIdx.x; j < n; j += QP_BLOCK_SIZE) {
    T a = c[j] < T(0) ? -c[j] : c[j];
    if (a > v) v = a;
  }
  T m = qp_block_max<T>(v, scratch);
  return m > T(1) ? m : T(1);
}

// ---------------------------------------------------------------------------
// Shared-memory layouts
// ---------------------------------------------------------------------------

template <typename T>
inline std::size_t qp_grad_smem_bytes(int n)
{
  // G[n*n] + c[n] + x[n] + g[n] + y[n] + xold[n] + scratch[N_WARPS]
  std::size_t bytes = 0;
  bytes += sizeof(T) * static_cast<std::size_t>(n) * n;  // G
  bytes += sizeof(T) * static_cast<std::size_t>(n) * 5;  // c, x, g, y, xold
  bytes += sizeof(T) * QP_N_WARPS;                       // scratch
  return bytes;
}

template <typename T>
inline std::size_t qp_lbfgs_smem_bytes(int n, int hist)
{
  // G[n*n] + c,x,g,xold,gold,q,d [7n] + s_hist,y_hist [2*hist*n]
  // + rho[hist] + alpha[hist] + scratch[N_WARPS]
  std::size_t bytes = 0;
  bytes += sizeof(T) * static_cast<std::size_t>(n) * n;
  bytes += sizeof(T) * static_cast<std::size_t>(n) * 7;
  bytes += sizeof(T) * static_cast<std::size_t>(n) * 2 * hist;
  bytes += sizeof(T) * static_cast<std::size_t>(hist) * 2;
  bytes += sizeof(T) * QP_N_WARPS;
  return bytes;
}

/** Copy resident G into shared memory and load c = C[:, p], init x = 0. */
template <typename T>
__device__ inline void qp_load(
  const T* G_global, const T* C, int n, int p, T* G, T* c, T* x)
{
  const T* c_p = C + static_cast<std::size_t>(p) * n;
  for (int q = threadIdx.x; q < n * n; q += QP_BLOCK_SIZE)
    G[q] = G_global[q];
  for (int j = threadIdx.x; j < n; j += QP_BLOCK_SIZE) {
    c[j] = c_p[j];
    x[j] = T(0);
  }
  __syncthreads();
}

// ---------------------------------------------------------------------------
// SGD: projected gradient descent (ISTA) with fixed step 1/L
// ---------------------------------------------------------------------------

template <typename T>
__global__ __launch_bounds__(QP_BLOCK_SIZE) void nnls_sgd_batched_kernel(
  const T* __restrict__ G_global,
  const T* __restrict__ C,
  int n,
  const std::uint8_t* __restrict__ masks,
  T* __restrict__ X,
  T inv_L,
  int max_iter,
  int check_every,
  T tol)
{
  extern __shared__ unsigned char smem_raw[];
  T* G   = reinterpret_cast<T*>(smem_raw);
  T* c   = G + static_cast<std::size_t>(n) * n;
  T* x   = c + n;
  T* g   = x + n;
  T* scratch = g + n;  // remaining y/xold slots unused here

  const int p = blockIdx.x;
  const std::uint8_t* mask_p =
    (masks == nullptr) ? nullptr : masks + static_cast<std::size_t>(p) * n;

  qp_load<T>(G_global, C, n, p, G, c, x);
  const T scale = qp_scale<T>(c, n, scratch);

  for (int k = 0; k < max_iter; ++k) {
    qp_gradient<T>(G, c, x, g, n);
    for (int j = threadIdx.x; j < n; j += QP_BLOCK_SIZE) {
      if (mask_p == nullptr || mask_p[j] != 0) {
        T v = x[j] - inv_L * g[j];
        x[j] = v > T(0) ? v : T(0);
      } else {
        x[j] = T(0);
      }
    }
    __syncthreads();

    if ((k % check_every) == 0) {
      T kkt = qp_kkt_residual<T>(x, g, mask_p, n, scratch);
      if (kkt < tol * scale) break;
    }
  }

  T* x_p = X + static_cast<std::size_t>(p) * n;
  for (int j = threadIdx.x; j < n; j += QP_BLOCK_SIZE)
    x_p[j] = x[j];
}

// ---------------------------------------------------------------------------
// APG: FISTA-style accelerated projected gradient with fixed step 1/L
// ---------------------------------------------------------------------------

template <typename T>
__global__ __launch_bounds__(QP_BLOCK_SIZE) void nnls_apg_batched_kernel(
  const T* __restrict__ G_global,
  const T* __restrict__ C,
  int n,
  const std::uint8_t* __restrict__ masks,
  T* __restrict__ X,
  T inv_L,
  int max_iter,
  int check_every,
  T tol)
{
  extern __shared__ unsigned char smem_raw[];
  T* G    = reinterpret_cast<T*>(smem_raw);
  T* c    = G + static_cast<std::size_t>(n) * n;
  T* x    = c + n;      // current iterate x_k
  T* g    = x + n;      // gradient at y
  T* y    = g + n;      // momentum point
  T* xold = y + n;      // x_{k-1}
  T* scratch = xold + n;

  const int p = blockIdx.x;
  const std::uint8_t* mask_p =
    (masks == nullptr) ? nullptr : masks + static_cast<std::size_t>(p) * n;

  qp_load<T>(G_global, C, n, p, G, c, x);
  const T scale = qp_scale<T>(c, n, scratch);
  for (int j = threadIdx.x; j < n; j += QP_BLOCK_SIZE) {
    y[j]    = x[j];
    xold[j] = x[j];
  }
  __syncthreads();

  // Nesterov scalar recurrence is deterministic, so every thread keeps an
  // identical copy in registers (no cross-thread communication needed).
  T t_curr = T(1);

  for (int k = 0; k < max_iter; ++k) {
    qp_gradient<T>(G, c, y, g, n);
    for (int j = threadIdx.x; j < n; j += QP_BLOCK_SIZE) {
      if (mask_p == nullptr || mask_p[j] != 0) {
        T v  = y[j] - inv_L * g[j];
        x[j] = v > T(0) ? v : T(0);
      } else {
        x[j] = T(0);
      }
    }
    __syncthreads();

    if ((k % check_every) == 0) {
      T kkt = qp_kkt_residual<T>(x, g, mask_p, n, scratch);
      if (kkt < tol * scale) break;
    }

    const T t_next = (T(1) + std::sqrt(T(1) + T(4) * t_curr * t_curr)) / T(2);
    const T beta   = (t_curr - T(1)) / t_next;
    for (int j = threadIdx.x; j < n; j += QP_BLOCK_SIZE) {
      T xn = x[j];
      y[j]    = xn + beta * (xn - xold[j]);
      xold[j] = xn;
    }
    __syncthreads();
    t_curr = t_next;
  }

  T* x_p = X + static_cast<std::size_t>(p) * n;
  for (int j = threadIdx.x; j < n; j += QP_BLOCK_SIZE)
    x_p[j] = x[j];
}

// ---------------------------------------------------------------------------
// CD: coordinate descent on the QP (Gauss-Seidel with incremental gradient)
// ---------------------------------------------------------------------------

template <typename T>
__global__ __launch_bounds__(QP_BLOCK_SIZE) void nnls_cd_batched_kernel(
  const T* __restrict__ G_global,
  const T* __restrict__ C,
  int n,
  const std::uint8_t* __restrict__ masks,
  T* __restrict__ X,
  int max_iter,
  int check_every,
  T tol)
{
  extern __shared__ unsigned char smem_raw[];
  T* G   = reinterpret_cast<T*>(smem_raw);
  T* c   = G + static_cast<std::size_t>(n) * n;
  T* x   = c + n;
  T* g   = x + n;      // maintained gradient g = G x - c
  T* scratch = g + n;

  const int p = blockIdx.x;
  const std::uint8_t* mask_p =
    (masks == nullptr) ? nullptr : masks + static_cast<std::size_t>(p) * n;

  qp_load<T>(G_global, C, n, p, G, c, x);
  const T scale = qp_scale<T>(c, n, scratch);
  // x == 0  =>  g = -c
  for (int j = threadIdx.x; j < n; j += QP_BLOCK_SIZE)
    g[j] = -c[j];
  __syncthreads();

  const T eps = (sizeof(T) == 4 ? T(1e-12) : T(1e-24));

  for (int sweep = 0; sweep < max_iter; ++sweep) {
    for (int j = 0; j < n; ++j) {
      if (mask_p != nullptr && mask_p[j] == 0) continue;
      const T gjj = G[j + j * n];
      if (!(gjj > eps)) continue;
      // Every thread computes the same delta from shared state.
      const T cand   = x[j] - g[j] / gjj;
      const T xj_new = cand > T(0) ? cand : T(0);
      const T delta  = xj_new - x[j];
      if (threadIdx.x == 0) x[j] = xj_new;
      __syncthreads();
      if (delta != T(0)) {
        for (int i = threadIdx.x; i < n; i += QP_BLOCK_SIZE)
          g[i] += G[i + j * n] * delta;
      }
      __syncthreads();
    }

    if ((sweep % check_every) == 0) {
      T kkt = qp_kkt_residual<T>(x, g, mask_p, n, scratch);
      if (kkt < tol * scale) break;
    }
  }

  T* x_p = X + static_cast<std::size_t>(p) * n;
  for (int j = threadIdx.x; j < n; j += QP_BLOCK_SIZE)
    x_p[j] = x[j];
}

// ---------------------------------------------------------------------------
// LBFGS: projected limited-memory BFGS on the QP, with a projected-gradient
// safeguard (guaranteed-descent fallback when the quasi-Newton step fails).
// ---------------------------------------------------------------------------

/** f(x) = 1/2 x^T G x - c^T x, computed from the current gradient g = Gx - c
 *  via  f = 1/2 x . (g - c). */
template <typename T>
__device__ inline T qp_objective(const T* x, const T* g, const T* c, int n, T* scratch)
{
  T v = T(0);
  for (int j = threadIdx.x; j < n; j += QP_BLOCK_SIZE)
    v += x[j] * (g[j] - c[j]);
  return T(0.5) * qp_block_sum<T>(v, scratch);
}

template <typename T>
__global__ __launch_bounds__(QP_BLOCK_SIZE) void nnls_lbfgs_batched_kernel(
  const T* __restrict__ G_global,
  const T* __restrict__ C,
  int n,
  const std::uint8_t* __restrict__ masks,
  T* __restrict__ X,
  T inv_L,
  int max_iter,
  int hist,
  int check_every,
  T tol)
{
  extern __shared__ unsigned char smem_raw[];
  T* G     = reinterpret_cast<T*>(smem_raw);
  T* c     = G + static_cast<std::size_t>(n) * n;
  T* x     = c + n;
  T* g     = x + n;
  T* xold  = g + n;
  T* gold  = xold + n;
  T* q     = gold + n;
  T* d     = q + n;
  T* s_hist = d + n;                                     // hist * n
  T* y_hist = s_hist + static_cast<std::size_t>(hist) * n;  // hist * n
  T* rho    = y_hist + static_cast<std::size_t>(hist) * n;  // hist
  T* alpha  = rho + hist;                                   // hist
  T* scratch = alpha + hist;

  const int p = blockIdx.x;
  const std::uint8_t* mask_p =
    (masks == nullptr) ? nullptr : masks + static_cast<std::size_t>(p) * n;

  qp_load<T>(G_global, C, n, p, G, c, x);
  const T scale = qp_scale<T>(c, n, scratch);

  qp_gradient<T>(G, c, x, g, n);

  int m_stored = 0;  // number of valid (s, y) pairs
  int head     = 0;  // index of most-recent slot
  T   gamma    = inv_L;

  for (int k = 0; k < max_iter; ++k) {
    T kkt = qp_kkt_residual<T>(x, g, mask_p, n, scratch);
    if (kkt < tol * scale) break;

    // ---- two-loop recursion:  d = -H_k * g  (masked coords excluded) ----
    for (int j = threadIdx.x; j < n; j += QP_BLOCK_SIZE)
      q[j] = (mask_p == nullptr || mask_p[j] != 0) ? g[j] : T(0);
    __syncthreads();

    for (int t = 0; t < m_stored; ++t) {
      int slot   = (head - t + hist) % hist;
      T*  s_i    = s_hist + static_cast<std::size_t>(slot) * n;
      T*  y_i    = y_hist + static_cast<std::size_t>(slot) * n;
      T   partial = T(0);
      for (int j = threadIdx.x; j < n; j += QP_BLOCK_SIZE)
        partial += s_i[j] * q[j];
      T sq = qp_block_sum<T>(partial, scratch);
      T a  = rho[slot] * sq;
      if (threadIdx.x == 0) alpha[slot] = a;
      for (int j = threadIdx.x; j < n; j += QP_BLOCK_SIZE)
        q[j] -= a * y_i[j];
      __syncthreads();
    }

    for (int j = threadIdx.x; j < n; j += QP_BLOCK_SIZE)
      d[j] = gamma * q[j];
    __syncthreads();

    for (int t = m_stored - 1; t >= 0; --t) {
      int slot = (head - t + hist) % hist;
      T*  s_i  = s_hist + static_cast<std::size_t>(slot) * n;
      T*  y_i  = y_hist + static_cast<std::size_t>(slot) * n;
      T   partial = T(0);
      for (int j = threadIdx.x; j < n; j += QP_BLOCK_SIZE)
        partial += y_i[j] * d[j];
      T yd   = qp_block_sum<T>(partial, scratch);
      T beta = rho[slot] * yd;
      T a    = alpha[slot];
      for (int j = threadIdx.x; j < n; j += QP_BLOCK_SIZE)
        d[j] += (a - beta) * s_i[j];
      __syncthreads();
    }

    // Search direction is -d.  Guard: if -d is not a descent direction
    // (g . (-d) >= 0) fall back to the projected gradient step.
    T gd_partial = T(0);
    for (int j = threadIdx.x; j < n; j += QP_BLOCK_SIZE)
      if (mask_p == nullptr || mask_p[j] != 0) gd_partial += g[j] * d[j];
    T gd = qp_block_sum<T>(gd_partial, scratch);
    bool use_lbfgs = gd > T(0);

    // Save current state for the (s, y) update.
    for (int j = threadIdx.x; j < n; j += QP_BLOCK_SIZE) {
      xold[j] = x[j];
      gold[j] = g[j];
    }
    __syncthreads();

    const T f0 = qp_objective<T>(x, g, c, n, scratch);

    // Projected backtracking line search along the chosen direction.
    T step = use_lbfgs ? T(1) : inv_L;
    bool accepted = false;
    for (int ls = 0; ls < 20; ++ls) {
      for (int j = threadIdx.x; j < n; j += QP_BLOCK_SIZE) {
        if (mask_p == nullptr || mask_p[j] != 0) {
          T dir = use_lbfgs ? -d[j] : -g[j];
          T v   = xold[j] + step * dir;
          x[j]  = v > T(0) ? v : T(0);
        } else {
          x[j] = T(0);
        }
      }
      __syncthreads();
      qp_gradient<T>(G, c, x, g, n);
      T f1 = qp_objective<T>(x, g, c, n, scratch);
      if (f1 < f0) {  // monotone sufficient-decrease for the convex QP
        accepted = true;
        break;
      }
      step *= T(0.5);
    }
    (void)accepted;  // if no step decreased f, keep the last projected point

    // ---- history update:  s = x - xold, y = g - gold ; store if s.y > 0 ----
    // Masked coordinates are pinned to 0 in the history vectors so the whole
    // quasi-Newton recursion stays confined to the active subspace.
    T sy_partial = T(0);
    T yy_partial = T(0);
    for (int j = threadIdx.x; j < n; j += QP_BLOCK_SIZE) {
      if (mask_p == nullptr || mask_p[j] != 0) {
        T sj = x[j] - xold[j];
        T yj = g[j] - gold[j];
        sy_partial += sj * yj;
        yy_partial += yj * yj;
      }
    }
    T sy = qp_block_sum<T>(sy_partial, scratch);
    T yy = qp_block_sum<T>(yy_partial, scratch);

    if (sy > T(1e-12) && yy > T(0)) {
      int slot = (head + 1) % hist;
      T*  s_i  = s_hist + static_cast<std::size_t>(slot) * n;
      T*  y_i  = y_hist + static_cast<std::size_t>(slot) * n;
      for (int j = threadIdx.x; j < n; j += QP_BLOCK_SIZE) {
        if (mask_p == nullptr || mask_p[j] != 0) {
          s_i[j] = x[j] - xold[j];
          y_i[j] = g[j] - gold[j];
        } else {
          s_i[j] = T(0);
          y_i[j] = T(0);
        }
      }
      if (threadIdx.x == 0) rho[slot] = T(1) / sy;
      __syncthreads();
      head     = slot;
      m_stored = m_stored < hist ? m_stored + 1 : hist;
      gamma    = sy / yy;
    }
  }

  T* x_p = X + static_cast<std::size_t>(p) * n;
  for (int j = threadIdx.x; j < n; j += QP_BLOCK_SIZE)
    x_p[j] = x[j];
}

}  // namespace detail
}  // namespace Solver
}  // namespace ML

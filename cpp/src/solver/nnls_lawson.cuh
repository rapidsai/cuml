/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/device_mdspan.hpp>              // device_matrix_view / col_major
#include <raft/core/error.hpp>                      // ASSERT
#include <raft/core/resource/cuda_stream.hpp>       // raft::resource::get_cuda_stream
#include <raft/core/resource/device_properties.hpp>  // raft::resource::get_device_properties
#include <raft/core/resources.hpp>                  // raft::resources
#include <raft/util/cuda_rt_essentials.hpp>         // RAFT_CUDA_TRY
#include <raft/util/cuda_utils.cuh>                 // raft::WarpSize
#include <raft/util/reduction.cuh>                  // raft::blockReduce / blockRankedReduce

#include <cuda_runtime.h>

#include <cstdint>
#include <optional>

namespace ML {
namespace Solver {
namespace detail {


/** Number of warps in a block of `BlockSize` threads (compile-time). */
constexpr int lawson_n_warps(int block_size) { return block_size / raft::WarpSize; }

/**
 * Reduced-precision storage type for the Cholesky working set that lives in
 * shared memory.  The Gram matrix G = A^T A is kept in global memory (read
 * directly, L2-cached across the grid), so the only n*n array in shared memory
 * is the incrementally-maintained Cholesky factor Gp, stored narrowed
 * (double -> float) to halve its footprint.  The O(n) solver state
 * (c, x, w, s, red_val) stays at full precision T; arithmetic still accumulates
 * in T.  For T == float this is the identity, so nothing changes.
 */
template <typename T>
struct Narrowing {
  using type = T;
};

template <>
struct Narrowing<double> {
  using type = float;
};

template <typename T>
using narrow_t = typename Narrowing<T>::type;

/**
 * Compute the dynamic shared-memory footprint of the Lawson-Hanson kernel for
 * a single problem with `n` columns and a block of `BlockSize` threads.  Layout
 * (T and int arrays first, then narrow_t<T>, then int8):
 *   T   c[n]                   A^T b
 *   T   x[n]                   current solution
 *   T   w[n]                   gradient (also scratch for the removed-index list)
 *   T   s[n]                   trial solution (also RHS / downdate scratch)
 *   T   red_val[WarpSize]      reduction scratch (also used to broadcast scalars)
 *   int red_idx[WarpSize]      reduction scratch (also used to broadcast scalars)
 *   int idx[n]                 compact list of active-set column indices
 *   narrow_t<T> Gp[n*n]        Cholesky factor L of the active submatrix
 *                              (leading dimension n), maintained incrementally
 *   int8 act[n]                1 if column is in active set, 0 otherwise
 *
 * red_val/red_idx back the RAFT block reductions (raft::blockRankedReduce needs
 * a value slot followed by an index slot per warp lane, i.e. WarpSize of each,
 * laid out contiguously with red_idx immediately after red_val); slot 0 of each
 * doubles as the scalar-broadcast channel.
 *
 * The Gram matrix G lives in global memory (read directly, L2-cached across the
 * grid) and is never staged into shared memory.  The single narrow_t<T> array
 * (Gp) would span 4*n*n bytes, which is not a multiple of alignof(double) for
 * odd n*n, so it is placed after the wider T/int arrays (which start 8-byte
 * aligned off the 16-byte-aligned smem base) and before the 1-byte act[]: its
 * own start offset is then always a multiple of 4, satisfying float alignment
 * without disturbing the double arrays.
 */
template <typename T, int BlockSize>
inline std::size_t lawson_smem_bytes(int n)
{
  std::size_t bytes = 0;
  bytes += sizeof(narrow_t<T>) * static_cast<std::size_t>(n) * n;  // Gp (factor L)
  bytes += sizeof(T) * static_cast<std::size_t>(n) * 4;            // c, x, w, s
  bytes += sizeof(T) * raft::WarpSize;                             // red_val
  bytes += sizeof(int) * raft::WarpSize;                          // red_idx
  bytes += sizeof(int) * static_cast<std::size_t>(n);            // idx
  bytes += sizeof(std::int8_t) * static_cast<std::size_t>(n);    // act
  return bytes;
}

template <typename T>
struct LawsonSmem {
  T* c;
  T* x;
  T* w;
  T* s;
  T* red_val;
  int* red_idx;
  int* idx;
  narrow_t<T>* Gp;
  std::int8_t* act;
};

template <typename T, int BlockSize>
__device__ LawsonSmem<T> lawson_smem_layout(unsigned char* smem, int n)
{
  LawsonSmem<T> L;
  // Wider arrays first (double is 8-byte aligned off the 16-byte-aligned base),
  // then the narrowed factor Gp, then the 1-byte act[]; see lawson_smem_bytes.
  L.c       = reinterpret_cast<T*>(smem);
  L.x       = L.c + n;
  L.w       = L.x + n;
  L.s       = L.w + n;
  L.red_val = L.s + n;
  // red_idx sits immediately after red_val so a single pointer (red_val) backs
  // raft::blockRankedReduce, which expects the index slots at &shbuf[WarpSize].
  L.red_idx = reinterpret_cast<int*>(L.red_val + raft::WarpSize);
  L.idx     = L.red_idx + raft::WarpSize;
  L.Gp      = reinterpret_cast<narrow_t<T>*>(L.idx + n);
  L.act     = reinterpret_cast<std::int8_t*>(L.Gp + n * n);
  return L;
}

/**
 * Block-wide argmax of `w[i]` restricted to indices where `act[i] == 0` and,
 * when `mask` is non-empty, column `i` is enabled by the support (`mask(i) != 0`).
 * Passing an empty `mask` view means every inactive column is eligible.  Returns
 * (max_value, argmax_index) on tid 0; the result is communicated to all threads
 * via the supplied `red_val[0]` / `red_idx[0]` slots.
 */
template <typename T, int BlockSize>
__device__ inline void block_argmax_inactive(
  const T*                                            w,
  const std::int8_t*                                  act,
  raft::device_vector_view<const std::uint8_t, int>   mask,
  int                                                 n,
  T*                                                  red_val,
  int*                                                red_idx)
{
  const int  tid      = threadIdx.x;
  const bool has_mask = mask.data_handle() != nullptr;

  T   thread_max = -std::numeric_limits<T>::infinity();
  int thread_idx = -1;
  for (int i = tid; i < n; i += BlockSize) {
    if (act[i] == 0 && (!has_mask || mask(i) != 0)) {
      T v = w[i];
      if (v > thread_max) {
        thread_max = v;
        thread_idx = i;
      }
    }
  }

  // red_val backs the reduction scratch (red_idx is contiguous right after it);
  // ties resolve to whichever lane RAFT keeps, which is fine for the algorithm.
  auto res = raft::blockRankedReduce(thread_max, red_val, thread_idx, raft::max_op{});
  if (tid == 0) {
    red_val[0] = res.first;
    red_idx[0] = res.second;
  }
  __syncthreads();
}

/**
 * Block-wide minimum of x[idx[jj]] / (x[idx[jj]] - s[jj]) over jj in [0, np)
 * where s[jj] <= 0.  Used to compute the alpha step.  Also returns the count
 * of binding indices (those with s[jj] <= 0) in `red_idx[0]`.  If no binding
 * indices are found, alpha is +inf.
 */
template <typename T, int BlockSize>
__device__ inline void block_min_alpha(
  const T* x, const T* s, const int* idx, int np, T* red_val, int* red_idx)
{
  const int tid = threadIdx.x;

  T   thread_min = std::numeric_limits<T>::infinity();
  int thread_cnt = 0;
  for (int jj = tid; jj < np; jj += BlockSize) {
    T s_jj = s[jj];
    if (s_jj <= T(0)) {
      T x_jj  = x[idx[jj]];
      T denom = x_jj - s_jj;  // strictly positive when x_jj >= 0 and s_jj <= 0
      // Numerical safety: if denom is tiny (x_jj == 0 and s_jj == 0) treat as
      // non-binding (no contribution).
      if (denom > T(0)) {
        T alpha = x_jj / denom;
        if (alpha < thread_min) thread_min = alpha;
        ++thread_cnt;
      }
    }
  }
  // Two block reductions over the same warp scratch, run back to back: the min
  // (blockRankedReduce pads absent lanes with +inf) and the binding count (a
  // plain sum, for which blockReduce's zero padding is the correct identity).
  auto min_res = raft::blockRankedReduce(thread_min, red_val, tid, raft::min_op{});
  int  n_bind  = raft::blockReduce<int>(thread_cnt, reinterpret_cast<char*>(red_idx), raft::add_op{});
  if (tid == 0) {
    red_val[0] = min_res.first;
    red_idx[0] = n_bind;
  }
  __syncthreads();
}

/**
 * Block-wide minimum of `s[0..np)`.  Result on `red_val[0]`.
 */
template <typename T, int BlockSize>
__device__ inline void block_min(const T* s, int np, T* red_val)
{
  const int tid = threadIdx.x;
  T thread_min  = std::numeric_limits<T>::infinity();
  for (int i = tid; i < np; i += BlockSize) {
    T v = s[i];
    if (v < thread_min) thread_min = v;
  }
  // red_val's WarpSize index scratch (red_idx) is reserved contiguously after
  // it, so blockRankedReduce is safe even though the index is unused here.
  auto res = raft::blockRankedReduce(thread_min, red_val, tid, raft::min_op{});
  if (tid == 0) red_val[0] = res.first;
  __syncthreads();
}

/**
 * Projected gradient w = c - G x, reading the Gram matrix G directly from global
 * memory (L2-cached across the grid).  Because x is zero outside the active set,
 * only the np active columns contribute:
 *   w[j] = c[j] - sum_{kk<np} G(j, idx[kk]) * x[idx[kk]].
 * For a fixed active column the threads stride over rows j, so consecutive lanes
 * read consecutive (column-major) elements of G -- a coalesced access.  np == 0
 * yields w = c.
 */
template <typename T, int BlockSize>
__device__ inline void block_matvec_gradient(
  T*                                                      w,
  const T*                                                c,
  raft::device_matrix_view<const T, int, raft::col_major> G,
  const int*                                              idx,
  const T*                                                x,
  int                                                     np,
  int                                                     n)
{
  const int tid = threadIdx.x;
  for (int j = tid; j < n; j += BlockSize) {
    T acc = c[j];
    for (int kk = 0; kk < np; ++kk) {
      const int k = idx[kk];
      acc -= G(j, k) * x[k];
    }
    w[j] = acc;
  }
  __syncthreads();
}

/**
 * Incremental "bordering" Cholesky update.  On entry L holds the (np-1)x(np-1)
 * lower factor of the previously active submatrix (column-major, leading
 * dimension `ld`); the newly activated column is idx[np-1].  The new Gram column
 * is read directly from global memory G and the factor is extended in place:
 *   solve L_11 l = a_12   for the new row l = L(np-1, 0:np-1),
 *   new diagonal          L(np-1, np-1) = sqrt(a_22 - l . l).
 * Returns false (leaving the leading (np-1) block untouched) when the new pivot
 * would be non-positive, i.e. activating idx[np-1] breaks positive-definiteness.
 *
 * Device analogue of raft::linalg::choleskyRank1Update, which is a host/cuBLAS
 * routine and so cannot be called from within a block.  `scratch` is O(np)
 * working space (the new column / row l).  The trace-based Tikhonov regulariser
 * of the from-scratch factorisation is replaced by a per-pivot guard on a_22.
 */
template <int BlockSize, typename GT, typename GG, typename T>
__device__ inline bool block_chol_append(
  GT*                                                      L,
  int                                                      ld,
  int                                                      np,
  raft::device_matrix_view<const GG, int, raft::col_major> G,
  const int*                                               idx,
  T*                                                       red_val,
  T*                                                       scratch)
{
  const int tid    = threadIdx.x;
  const int m      = np - 1;  // size of the existing factor L_11
  const int j_star = idx[np - 1];

  // a_12[i] = G(idx[i], j_star)  (read from global memory).
  for (int i = tid; i < m; i += BlockSize)
    scratch[i] = static_cast<T>(G(idx[i], j_star));
  __syncthreads();

  // Forward solve L_11 l = a_12, in place in scratch.
  for (int i = 0; i < m; ++i) {
    if (tid == 0) {
      red_val[0] = scratch[i] / static_cast<T>(L[i + i * ld]);
      scratch[i] = red_val[0];
    }
    __syncthreads();
    T y_i = red_val[0];
    for (int j = i + 1 + tid; j < m; j += BlockSize)
      scratch[j] -= static_cast<T>(L[j + i * ld]) * y_i;
    __syncthreads();
  }

  // Store l as the new row (np-1) of L and accumulate dot = l . l.
  T thread_dot = T(0);
  for (int j = tid; j < m; j += BlockSize) {
    T lj                 = scratch[j];
    L[(np - 1) + j * ld] = static_cast<GT>(lj);
    thread_dot += lj * lj;
  }
  T dot = raft::blockReduce<T>(thread_dot, reinterpret_cast<char*>(red_val), raft::add_op{});
  if (tid == 0) red_val[0] = dot;
  __syncthreads();
  dot = red_val[0];

  // New diagonal L_22 = sqrt(a_22 + eps - dot); reject a non-positive pivot.
  if (tid == 0) {
    T a22 = static_cast<T>(G(j_star, j_star));
    T eps = (sizeof(GT) == 4 ? T(1e-7) : T(1e-14)) * (a22 > T(0) ? a22 : T(1));
    T d2  = a22 + eps - dot;
    if (d2 > T(0)) {
      L[(np - 1) + (np - 1) * ld] = static_cast<GT>(std::sqrt(d2));
      red_val[0]                  = T(1);
    } else {
      red_val[0] = T(-1);
    }
  }
  __syncthreads();
  return red_val[0] > T(0);
}

/**
 * Remove active-set position `p` (0-based, in the current np-ordering) from the
 * np x np lower Cholesky factor L (column-major, leading dimension `ld`),
 * producing the (np-1)x(np-1) factor of the submatrix with row/column p deleted.
 *
 * Deleting an interior row/column reduces to a positive rank-1 Cholesky update
 * of the trailing diagonal block by the below-diagonal part of column p:
 *   M M^T = L33 L33^T + l3p l3p^T,
 * applied with a sequence of Givens rotations (Golub & Van Loan, "deleting a
 * column").  `v` is O(np) scratch holding l3p and the rotated residual.
 */
template <int BlockSize, typename GT, typename T>
__device__ inline void block_chol_delete_one(GT* L, int ld, int np, int p, T* v)
{
  __shared__ T rot[2];  // (c, s) rotation, broadcast from thread 0
  const int tid = threadIdx.x;
  const int q   = np - 1 - p;  // trailing block size

  // Save l3p = L(p+1 : np-1, p) before the compaction overwrites column p.
  for (int i = tid; i < q; i += BlockSize)
    v[i] = static_cast<T>(L[(p + 1 + i) + p * ld]);
  __syncthreads();

  // Compact: drop row p and column p.  The per-column row shifts move each
  // destination from a not-yet-written source, so thread 0 performs them
  // race-free (O(np^2) with small np); columns < p keep rows [0, p) in place.
  if (tid == 0) {
    for (int j = 0; j < np - 1; ++j) {
      if (j < p) {
        for (int i = p; i < np - 1; ++i)
          L[i + j * ld] = L[(i + 1) + j * ld];
      } else {
        const int c = j + 1;
        for (int i = j; i < np - 1; ++i)
          L[i + j * ld] = L[(i + 1) + c * ld];
      }
    }
  }
  __syncthreads();

  // Positive rank-1 update of the trailing block (rows/cols p..np-2) by v.
  for (int k = 0; k < q; ++k) {
    if (tid == 0) {
      T Lkk  = static_cast<T>(L[(p + k) + (p + k) * ld]);
      T vk   = v[k];
      T r    = std::sqrt(Lkk * Lkk + vk * vk);
      rot[0] = (r > T(0)) ? (Lkk / r) : T(1);  // c
      rot[1] = (r > T(0)) ? (vk / r) : T(0);   // s
      L[(p + k) + (p + k) * ld] = static_cast<GT>(r);
    }
    __syncthreads();
    const T c = rot[0];
    const T s = rot[1];
    for (int t = k + 1 + tid; t < q; t += BlockSize) {
      const int row = p + t;
      T lik = static_cast<T>(L[row + (p + k) * ld]);
      T vi  = v[t];
      L[row + (p + k) * ld] = static_cast<GT>(c * lik + s * vi);
      v[t]                  = c * vi - s * lik;
    }
    __syncthreads();
  }
}

/**
 * Forward + back substitution for the system L L^T s = s_rhs.  The right-hand
 * side is provided in `s` and overwritten with the solution.  L is the lower
 * triangular factor stored column-major with leading dimension `ld`; only the
 * np x np leading block (lower triangle) is read, so the incrementally-updated
 * factor (fixed ld = n, current size np) can be solved without repacking.
 *
 * Both passes are sequential in the row index but parallel in the trailing
 * update, which is the standard cooperative pattern for small triangular solves.
 */
template <int BlockSize, typename GT, typename T>
__device__ inline void block_chol_solve(const GT* L, int ld, int np, T* s, T* red_val)
{
  const int tid = threadIdx.x;

  // L is stored narrowed (GT); the RHS/solution `s` stays at full precision T,
  // so factors are widened to T on read and the solve accumulates in T.

  // Forward solve: L y = s_rhs  ->  s holds y on exit.
  for (int i = 0; i < np; ++i) {
    if (tid == 0) {
      red_val[0] = s[i] / static_cast<T>(L[i + i * ld]);
      s[i]       = red_val[0];
    }
    __syncthreads();
    T y_i = red_val[0];
    for (int j = i + 1 + tid; j < np; j += BlockSize)
      s[j] -= static_cast<T>(L[j + i * ld]) * y_i;
    __syncthreads();
  }

  // Back solve: L^T x = y  ->  s holds x on exit.
  for (int i = np - 1; i >= 0; --i) {
    if (tid == 0) {
      red_val[0] = s[i] / static_cast<T>(L[i + i * ld]);
      s[i]       = red_val[0];
    }
    __syncthreads();
    T x_i = red_val[0];
    for (int j = tid; j < i; j += BlockSize)
      s[j] -= static_cast<T>(L[i + j * ld]) * x_i;  // L^T[i,j] = L[j,i]; rows < i
    __syncthreads();
  }
}

/**
 * Batched, masked Lawson-Hanson NNLS kernel -- the single solver kernel used
 * for both batched and single-problem solves.  One CUDA block solves problem
 * `p = blockIdx.x`, reading the Gram matrix `G = A^T A` directly from global
 * memory (shared by the whole grid, L2-cached) and its own RHS projection from
 * column `p` of `C = A^T B` and active-column support from column `p` of `masks`.
 *
 * The active-set Cholesky factor is maintained incrementally: each outer
 * iteration appends the entering column with a bordering update
 * (block_chol_append), and the inner line search shrinks it with Givens
 * downdates (block_chol_delete_one).  Consequently G is read only once per outer
 * iteration -- the active columns for the projected gradient plus the single new
 * column for the append -- and never re-gathered inside the inner loop.
 *
 * A "non-batched" solve is simply the P == 1, empty-`masks` case: the caller
 * forms G and C = A^T b once with cuBLAS and launches this kernel with a single
 * block (see nnls_batched_impl).  Columns disabled by a problem's mask are
 * excluded from the argmax so they can never enter the solution.
 *
 * @tparam BlockSize  number of threads per block (a multiple of raft::WarpSize;
 *                    chosen at launch by LawsonBlockDispatch).
 * @param G      (n, n) Gram matrix (column-major), shared by all problems.
 * @param C      (n, P) matrix of A^T B (column-major).
 * @param masks  (n, P) column-major uint8 support; column p is problem p's
 *               support.  May be an empty view, meaning "all columns eligible".
 * @param X      (n, P) solutions (column-major); written on exit.
 * @param max_iter  outer-iteration cap.
 * @param tol       optimality tolerance on the projected gradient.
 */
template <typename T, int BlockSize>
__global__ __launch_bounds__(BlockSize) void nnls_lawson_batched_kernel(
  raft::device_matrix_view<const T, int, raft::col_major>            G,
  raft::device_matrix_view<const T, int, raft::col_major>            C,
  raft::device_matrix_view<const std::uint8_t, int, raft::col_major> masks,
  raft::device_matrix_view<T, int, raft::col_major>                  X,
  int max_iter,
  T tol)
{
  const int n = G.extent(0);
  const int p = blockIdx.x;

  extern __shared__ unsigned char smem_raw[];
  LawsonSmem<T> S = lawson_smem_layout<T, BlockSize>(smem_raw, n);
  // The active-set Cholesky factor L is stored narrowed in S.Gp with a fixed
  // leading dimension n, so incremental append/downdate never repack it.
  narrow_t<T>* L = S.Gp;

  __shared__ int sm_n_active;
  __shared__ int sm_j_star;
  __shared__ int sm_new_n;
  __shared__ int sm_n_removed;

  const int tid = threadIdx.x;

  // Column p of the (possibly empty) support, as a 1-D view for the argmax.
  auto mask_col = (masks.size() != 0)
                    ? raft::make_device_vector_view<const std::uint8_t, int>(&masks(0, p), n)
                    : raft::device_vector_view<const std::uint8_t, int>{};

  // ---- Phase 1+2: load c = C[:, p]; init x and active set ------------------
  for (int j = tid; j < n; j += BlockSize) {
    S.c[j]   = C(j, p);
    S.x[j]   = T(0);
    S.act[j] = 0;
  }
  if (tid == 0) sm_n_active = 0;
  __syncthreads();

  const int inner_budget_total = 3 * n + 1;

  // ---- Phase 3: outer loop (active-set growth) -----------------------------
  for (int outer = 0; outer < max_iter; ++outer) {
    // Projected gradient w = c - G x, reading active columns of G from global.
    block_matvec_gradient<T, BlockSize>(S.w, S.c, G, S.idx, S.x, sm_n_active, n);

    block_argmax_inactive<T, BlockSize>(S.w, S.act, mask_col, n, S.red_val, S.red_idx);
    T   max_w  = S.red_val[0];
    int j_star = S.red_idx[0];
    if (j_star < 0 || max_w <= tol) break;

    // Activate j_star (append it at the end of the compact active set).
    if (tid == 0) {
      S.act[j_star]      = 1;
      S.idx[sm_n_active] = j_star;
      sm_n_active        = sm_n_active + 1;
      sm_j_star          = j_star;
    }
    __syncthreads();

    // Incremental bordering append: extend L with the new column read from
    // global G.  A non-positive pivot means the activation is rejected -> undo
    // it and stop the outer loop (matches the old Cholesky-failure behaviour).
    bool ok = block_chol_append<BlockSize>(L, n, sm_n_active, G, S.idx, S.red_val, S.s);
    if (!ok) {
      if (tid == 0) {
        sm_n_active      = sm_n_active - 1;
        S.act[sm_j_star] = 0;
      }
      __syncthreads();
      break;
    }

    for (int inner = 0; inner < inner_budget_total; ++inner) {
      const int np = sm_n_active;

      // RHS c_P = c[idx]; solve L L^T s = c_P on the current factor.
      for (int jj = tid; jj < np; jj += BlockSize)
        S.s[jj] = S.c[S.idx[jj]];
      __syncthreads();

      block_chol_solve<BlockSize>(L, n, np, S.s, S.red_val);

      block_min<T, BlockSize>(S.s, np, S.red_val);
      T min_s = S.red_val[0];
      if (min_s > T(0)) {
        for (int j = tid; j < n; j += BlockSize)
          S.x[j] = T(0);
        __syncthreads();
        for (int jj = tid; jj < np; jj += BlockSize)
          S.x[S.idx[jj]] = S.s[jj];
        __syncthreads();
        break;
      }

      block_min_alpha<T, BlockSize>(S.x, S.s, S.idx, np, S.red_val, S.red_idx);
      T   alpha     = S.red_val[0];
      int n_binding = S.red_idx[0];
      if (n_binding == 0) break;

      for (int jj = tid; jj < np; jj += BlockSize) {
        int j_idx  = S.idx[jj];
        T   xi     = S.x[j_idx];
        T   si     = S.s[jj];
        S.x[j_idx] = xi + alpha * (si - xi);
      }
      __syncthreads();

      // Compact the active set, recording the removed local positions (ascending)
      // in the free scratch backed by S.w (unused inside the inner loop).
      int* rem = reinterpret_cast<int*>(S.w);
      if (tid == 0) {
        const T zero_eps = (sizeof(T) == 4 ? T(1e-12) : T(1e-15));
        int new_n        = 0;
        int n_rem        = 0;
        for (int jj = 0; jj < np; ++jj) {
          int j_idx = S.idx[jj];
          if (S.x[j_idx] > zero_eps) {
            S.idx[new_n++] = j_idx;
          } else {
            S.act[j_idx] = 0;
            S.x[j_idx]   = T(0);
            rem[n_rem++] = jj;
          }
        }
        sm_new_n     = new_n;
        sm_n_removed = n_rem;
      }
      __syncthreads();

      // Downdate L for each removed position, deleting in descending order so
      // earlier (lower-index) deletions stay valid as np shrinks.
      int cur_np = np;
      for (int r = sm_n_removed - 1; r >= 0; --r) {
        block_chol_delete_one<BlockSize>(L, n, cur_np, rem[r], S.s);
        --cur_np;
      }
      if (tid == 0) sm_n_active = sm_new_n;
      __syncthreads();

      if (sm_n_active == 0) break;
    }
  }

  for (int j = tid; j < n; j += BlockSize)
    X(j, p) = S.x[j];
}

/**
 * Raise a kernel's dynamic-shared-memory cap when the requested carveout
 * exceeds the default 48 KB (SM 80/86/89/90 support up to 96 KB+).  Shared by
 * every batched NNLS backend launcher.
 */
template <typename Kernel>
inline void nnls_set_smem_attr(Kernel kernel, std::size_t smem_bytes)
{
  constexpr std::size_t kDefaultSmem = 48 * 1024;
  constexpr std::size_t kMaxSmem     = 96 * 1024;
  ASSERT(smem_bytes <= kMaxSmem,
         "ML::Solver::nnlsBatched: required shared memory (%zu B) exceeds the "
         "per-block limit (%zu B); reduce n_cols or pick a different solver.",
         smem_bytes,
         kMaxSmem);
  if (smem_bytes > kDefaultSmem) {
    cudaError_t err = cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes));
    ASSERT(err == cudaSuccess,
           "ML::Solver::nnlsBatched: cudaFuncSetAttribute failed: %s",
           cudaGetErrorString(err));
  }
}

// Desired number of co-resident blocks per problem before we keep a larger
// (lower-occupancy) block size instead of shrinking it.
constexpr int kResidenceMultiple = 8;

/**
 * Occupancy-driven block-size selector for the batched Lawson kernel.
 *
 * Starting from the largest block size, we ask the driver how many blocks of
 * this kernel instantiation can be co-resident across the whole GPU
 * (R = max active blocks per SM * number of SMs).  If `R * kResidenceMultiple`
 * covers the batch (>= n_problems) the grid can saturate the device at this
 * block size, so we launch it; otherwise we consider halving the block size --
 * which usually raises the block count -- and retry.
 *
 * The halving is gated by a second condition: we only shrink if it does not
 * lower the per-SM occupancy, measured as resident threads
 * (blocks_per_sm * BlockSize).  Near a hardware blocks-per-SM cap, or when the
 * kernel is shared-memory bound, a smaller block can fail to raise the block
 * count enough to compensate for the fewer threads each block carries, which
 * would trade device saturation for lower utilisation -- so in that case we
 * keep the larger block.  The recursion bottoms out at a single warp
 * (raft::WarpSize), where we always launch.
 */
template <typename T, int BlockSize = 32 * raft::WarpSize>
struct LawsonBlockDispatch {
  using GView = raft::device_matrix_view<const T, int, raft::col_major>;
  using XView = raft::device_matrix_view<T, int, raft::col_major>;
  using MView = raft::device_matrix_view<const std::uint8_t, int, raft::col_major>;

  /** Entry point: query the largest block's occupancy once, then recurse. */
  static void start(raft::resources const& handle,
                    GView                  G,
                    GView                  C,
                    std::optional<MView>   masks,
                    XView                  X,
                    int                    max_iter,
                    T                      tol)
  {
    const std::size_t smem = lawson_smem_bytes<T, BlockSize>(X.extent(0));
    // The occupancy query returns 0 for a carveout above the 48 KB default
    // unless the kernel's max-dynamic-smem attribute is raised first.
    nnls_set_smem_attr(nnls_lawson_batched_kernel<T, BlockSize>, smem);
    int blocks_per_sm = 0;
    RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, nnls_lawson_batched_kernel<T, BlockSize>, BlockSize, smem));
    run(handle, G, C, masks, X, max_iter, tol, blocks_per_sm);
  }

  // `blocks_per_sm` is the occupancy of this instantiation, already measured by
  // the previous recursion level (or by start()), so each level performs at
  // most one new occupancy query -- the one for the half-sized candidate.
  static void run(raft::resources const& handle,
                  GView                  G,
                  GView                  C,
                  std::optional<MView>   masks,
                  XView                  X,
                  int                    max_iter,
                  T                      tol,
                  int                    blocks_per_sm)
  {
    const int         n          = X.extent(0);
    const int         n_problems = X.extent(1);
    cudaStream_t      stream     = raft::resource::get_cuda_stream(handle);
    const std::size_t smem       = lawson_smem_bytes<T, BlockSize>(n);

    if constexpr (BlockSize > raft::WarpSize) {
      const int       n_sm     = raft::resource::get_device_properties(handle).multiProcessorCount;
      const long long resident = static_cast<long long>(blocks_per_sm) * n_sm;
      // RAFT_LOG_INFO("Resident: %lld, n_problems: %d", resident, n_problems);
      if (resident * kResidenceMultiple < n_problems) {
        // Shrink only if the smaller block keeps at least the same per-SM
        // occupancy (resident threads); otherwise we'd trade saturation for a
        // less utilised SM.  Reuse this query as the next level's occupancy.
        constexpr int     half_block = BlockSize / 2;
        const std::size_t smem_half  = lawson_smem_bytes<T, half_block>(n);
        // Raise the smem cap before querying; otherwise the driver reports 0
        // active blocks for any carveout above the 48 KB default.
        nnls_set_smem_attr(nnls_lawson_batched_kernel<T, half_block>, smem_half);
        int blocks_per_sm_half = 0;
        RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks_per_sm_half, nnls_lawson_batched_kernel<T, half_block>, half_block, smem_half));
        // RAFT_LOG_INFO("Consider block size: %d, occupancy: %d", half_block, blocks_per_sm_half);
        const int occ_full = blocks_per_sm * BlockSize;
        const int occ_half = blocks_per_sm_half * half_block;
        if ((occ_half >= occ_full) || (half_block >= n * 2)) {
          LawsonBlockDispatch<T, half_block>::run(
            handle, G, C, masks, X, max_iter, tol, blocks_per_sm_half);
          return;
        }
      }
    }

    // The smem cap for this instantiation was already raised exactly once, at
    // the point its occupancy was queried (in start() for the top block, or in
    // the parent level's half-block query), so we must not call the expensive
    // cudaFuncSetAttribute again here.
    // Empty view => "all columns eligible" inside the kernel.
    MView mv = masks.has_value() ? *masks : MView{};
    // RAFT_LOG_INFO("Selected block size: %d, smem: %zu, sizeof(T): %zu", BlockSize, smem, sizeof(T));
    nnls_lawson_batched_kernel<T, BlockSize>
      <<<n_problems, BlockSize, smem, stream>>>(G, C, mv, X, max_iter, tol);
  }
};

/** Dispatch a batched Lawson solve, choosing the block size from the kernel's
 *  occupancy and the batch size (see LawsonBlockDispatch). */
template <typename T>
inline void nnls_lawson_batched_dispatch(
  raft::resources const&                                               handle,
  raft::device_matrix_view<const T, int, raft::col_major>              G,
  raft::device_matrix_view<const T, int, raft::col_major>              C,
  std::optional<raft::device_matrix_view<const std::uint8_t, int, raft::col_major>> masks,
  raft::device_matrix_view<T, int, raft::col_major>                    X,
  int                                                                  max_iter,
  T                                                                    tol)
{
  LawsonBlockDispatch<T>::start(handle, G, C, masks, X, max_iter, tol);
}

}  // namespace detail
}  // namespace Solver
}  // namespace ML

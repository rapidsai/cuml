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
 * Reduced-precision storage type for the Gram / Cholesky working set that lives
 * in shared memory.  G = A^T A and its per-iteration Cholesky copy Gp are the
 * two n*n arrays that dominate the shared-memory footprint, so they are stored
 * narrowed (double -> float) to roughly halve it.  The O(n) solver state
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
 * (narrow_t<T> arrays first, then T arrays, then int, then int8):
 *   narrow_t<T> G[n*n]         Gram matrix (read-only after precompute)
 *   narrow_t<T> Gp[n*n]        working copy / Cholesky factor of the active submatrix
 *   T   c[n]                   A^T b
 *   T   x[n]                   current solution
 *   T   w[n]                   gradient
 *   T   s[n]                   trial solution (also used as RHS during tri-solve)
 *   T   red_val[WarpSize]      reduction scratch (also used to broadcast scalars)
 *   int red_idx[WarpSize]      reduction scratch (also used to broadcast scalars)
 *   int idx[n]                 compact list of active-set column indices
 *   int8 act[n]                1 if column is in active set, 0 otherwise
 *
 * red_val/red_idx back the RAFT block reductions (raft::blockRankedReduce needs
 * a value slot followed by an index slot per warp lane, i.e. WarpSize of each,
 * laid out contiguously with red_idx immediately after red_val); slot 0 of each
 * doubles as the scalar-broadcast channel.
 *
 * The narrow_t<T> block spans 2*n*n elements = 8*n*n bytes (float), which is a
 * multiple of alignof(double), so the following T block stays aligned.
 */
template <typename T, int BlockSize>
inline std::size_t lawson_smem_bytes(int n)
{
  std::size_t bytes = 0;
  bytes += sizeof(narrow_t<T>) * static_cast<std::size_t>(n) * n;  // G
  bytes += sizeof(narrow_t<T>) * static_cast<std::size_t>(n) * n;  // Gp
  bytes += sizeof(T) * static_cast<std::size_t>(n) * 4;            // c, x, w, s
  bytes += sizeof(T) * raft::WarpSize;                             // red_val
  bytes += sizeof(int) * raft::WarpSize;                          // red_idx
  bytes += sizeof(int) * static_cast<std::size_t>(n);            // idx
  bytes += sizeof(std::int8_t) * static_cast<std::size_t>(n);    // act
  return bytes;
}

template <typename T>
struct LawsonSmem {
  narrow_t<T>* G;
  narrow_t<T>* Gp;
  T* c;
  T* x;
  T* w;
  T* s;
  T* red_val;
  int* idx;
  int* red_idx;
  std::int8_t* act;
};

template <typename T, int BlockSize>
__device__ LawsonSmem<T> lawson_smem_layout(unsigned char* smem, int n)
{
  LawsonSmem<T> L;
  L.G       = reinterpret_cast<narrow_t<T>*>(smem);
  L.Gp      = L.G + n * n;
  L.c       = reinterpret_cast<T*>(L.Gp + n * n);
  L.x       = L.c + n;
  L.w       = L.x + n;
  L.s       = L.w + n;
  L.red_val = L.s + n;
  // red_idx sits immediately after red_val so a single pointer (red_val) backs
  // raft::blockRankedReduce, which expects the index slots at &shbuf[WarpSize].
  L.red_idx = reinterpret_cast<int*>(L.red_val + raft::WarpSize);
  L.idx     = L.red_idx + raft::WarpSize;
  L.act     = reinterpret_cast<std::int8_t*>(L.idx + n);
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
 * In-place right-looking Cholesky factorisation Gp = L L^T (L lower triangular)
 * for a column-major np x np matrix view (np = Gp.extent(0)) stored in shared
 * memory.  Each block cooperates; thread 0 handles the diagonal sqrt, then
 * threads share the column scaling and outer-product update.  Returns true on
 * success and false if a non-positive pivot was encountered.  The check is
 * performed by thread 0 and broadcast via the supplied `red_val` slot.
 *
 * Numerical safeguard: a tiny multiple of the trace is added to the diagonal
 * before factorisation to keep the system well-conditioned even when Gp is
 * near-singular.  This is a Tikhonov regulariser of magnitude
 * O(eps * trace(Gp)).
 */
template <int BlockSize, typename GT, typename T>
__device__ inline bool block_cholesky(
  raft::device_matrix_view<GT, int, raft::col_major> Gp, T* red_val)
{
  const int tid = threadIdx.x;
  const int np  = Gp.extent(0);

  // Compute trace and add eps_diag * trace / np to each diagonal entry.  The
  // matrix is stored narrowed (GT), but we accumulate/compute in the wider
  // scalar type T and only narrow on store.
  T thread_tr = T(0);
  for (int k = tid; k < np; k += BlockSize)
    thread_tr += static_cast<T>(Gp(k, k));
  T trace = raft::blockReduce<T>(thread_tr, reinterpret_cast<char*>(red_val), raft::add_op{});
  if (tid == 0) red_val[0] = trace;
  __syncthreads();
  trace = red_val[0];
  // Regularise at the precision actually stored (GT), not the scalar type T.
  T eps   = (sizeof(GT) == 4 ? T(1e-7) : T(1e-14)) * (trace > T(0) ? trace / T(np) : T(1));
  for (int k = tid; k < np; k += BlockSize)
    Gp(k, k) = static_cast<GT>(static_cast<T>(Gp(k, k)) + eps);
  __syncthreads();

  for (int k = 0; k < np; ++k) {
    if (tid == 0) {
      T d        = static_cast<T>(Gp(k, k));
      bool ok    = d > T(0);
      red_val[0] = ok ? std::sqrt(d) : T(-1);
      Gp(k, k)   = static_cast<GT>(red_val[0]);
    }
    __syncthreads();
    T diag = red_val[0];
    if (!(diag > T(0))) return false;

    // Column scale: Gp(i, k) /= diag for i > k
    for (int i = k + 1 + tid; i < np; i += BlockSize)
      Gp(i, k) = static_cast<GT>(static_cast<T>(Gp(i, k)) / diag);
    __syncthreads();

    // Outer-product update: Gp(i, j) -= Gp(i, k) * Gp(j, k) for j > k, i >= j
    int sub = np - k - 1;
    if (sub > 0) {
      int total = sub * sub;  // we'll use just lower triangle i >= j
      for (int q = tid; q < total; q += BlockSize) {
        int jj = q / sub;  // j relative to k+1
        int ii = q % sub;  // i relative to k+1
        if (ii >= jj) {
          int i  = k + 1 + ii;
          int j  = k + 1 + jj;
          T lik  = static_cast<T>(Gp(i, k));
          T ljk  = static_cast<T>(Gp(j, k));
          Gp(i, j) = static_cast<GT>(static_cast<T>(Gp(i, j)) - lik * ljk);
        }
      }
      __syncthreads();
    }
  }
  return true;
}

/**
 * Forward + back substitution for the system L L^T s = s_rhs.  The right-hand
 * side is provided in `s` and overwritten with the solution.  L is the lower
 * triangular factor produced by block_cholesky (column-major, np x np).
 *
 * Both passes are sequential in the row index but parallel in the trailing
 * update, which is the standard cooperative pattern for small triangular
 * solves.  `L` is the np x np (np = L.extent(0)) column-major factor.
 */
template <int BlockSize, typename GT, typename T>
__device__ inline void block_chol_solve(
  raft::device_matrix_view<GT, int, raft::col_major> L, T* s, T* red_val)
{
  const int tid = threadIdx.x;
  const int np  = L.extent(0);

  // L is stored narrowed (GT); the RHS/solution `s` stays at full precision T,
  // so factors are widened to T on read and the solve accumulates in T.

  // Forward solve: L y = s_rhs  ->  s holds y on exit.
  for (int i = 0; i < np; ++i) {
    if (tid == 0) {
      red_val[0] = s[i] / static_cast<T>(L(i, i));
      s[i] = red_val[0];
    }
    __syncthreads();
    T y_i = red_val[0];
    for (int j = i + 1 + tid; j < np; j += BlockSize)
      s[j] -= static_cast<T>(L(j, i)) * y_i;
    __syncthreads();
  }

  // Back solve: L^T x = y  ->  s holds x on exit.
  for (int i = np - 1; i >= 0; --i) {
    if (tid == 0) {
      red_val[0] = s[i] / static_cast<T>(L(i, i));
      s[i] = red_val[0];
    }
    __syncthreads();
    T x_i = red_val[0];
    for (int j = tid; j < i; j += BlockSize)
      s[j] -= static_cast<T>(L(i, j)) * x_i;  // L^T[i,j] = L[j,i]; rows < i
    __syncthreads();
  }
}

/**
 * Batched, masked Lawson-Hanson NNLS kernel -- the single solver kernel used
 * for both batched and single-problem solves.  One CUDA block solves problem
 * `p = blockIdx.x`, sharing the resident Gram matrix `G = A^T A` across the
 * whole grid and reading its own RHS projection from column `p` of `C = A^T B`
 * and its own active-column support from column `p` of `masks`.
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
  // Shared Gram wrapped as a column-major view so the active-set gather and the
  // projected-gradient product read it via operator() instead of hand-rolled
  // column-major offsets.  Stored narrowed (narrow_t<T>) to save shared memory.
  auto Gs = raft::make_device_matrix_view<narrow_t<T>, int, raft::col_major>(S.G, n, n);

  __shared__ int sm_n_active;
  __shared__ int sm_j_star;
  __shared__ int sm_should_break_outer;

  const int tid = threadIdx.x;

  // Column p of the (possibly empty) support, as a 1-D view for the argmax.
  auto mask_col = (masks.size() != 0)
                    ? raft::make_device_vector_view<const std::uint8_t, int>(&masks(0, p), n)
                    : raft::device_vector_view<const std::uint8_t, int>{};

  // ---- Phase 1+2: load resident G, c = C[:, p]; init x and active set ------
  for (int q = tid; q < n * n; q += BlockSize)
    S.G[q] = static_cast<narrow_t<T>>(G.data_handle()[q]);  // narrow into smem
  for (int j = tid; j < n; j += BlockSize) {
    S.c[j]   = C(j, p);
    S.x[j]   = T(0);
    S.act[j] = 0;
  }
  if (tid == 0) {
    sm_n_active           = 0;
    sm_should_break_outer = 0;
  }
  __syncthreads();

  const int inner_budget_total = 3 * n + 1;

  // ---- Phase 3: outer loop (active-set growth) -----------------------------
  for (int outer = 0; outer < max_iter; ++outer) {
    for (int j = tid; j < n; j += BlockSize) {
      T acc = S.c[j];
      for (int k = 0; k < n; ++k)
        acc -= static_cast<T>(Gs(j, k)) * S.x[k];
      S.w[j] = acc;
    }
    __syncthreads();

    block_argmax_inactive<T, BlockSize>(S.w, S.act, mask_col, n, S.red_val, S.red_idx);
    T   max_w  = S.red_val[0];
    int j_star = S.red_idx[0];
    if (j_star < 0 || max_w <= tol) break;

    if (tid == 0) {
      S.act[j_star]      = 1;
      S.idx[sm_n_active] = j_star;
      sm_n_active        = sm_n_active + 1;
      sm_j_star          = j_star;
    }
    __syncthreads();

    for (int inner = 0; inner < inner_budget_total; ++inner) {
      const int np = sm_n_active;
      auto Gps = raft::make_device_matrix_view<narrow_t<T>, int, raft::col_major>(S.Gp, np, np);

      for (int q = tid; q < np * np; q += BlockSize) {
        int jj = q / np;
        int ii = q % np;
        Gps(ii, jj) = Gs(S.idx[ii], S.idx[jj]);
      }
      for (int jj = tid; jj < np; jj += BlockSize)
        S.s[jj] = S.c[S.idx[jj]];
      __syncthreads();

      bool ok = block_cholesky<BlockSize>(Gps, S.red_val);
      if (!ok) {
        if (tid == 0) {
          sm_n_active           = sm_n_active - 1;
          S.act[sm_j_star]      = 0;
          sm_should_break_outer = 1;
        }
        __syncthreads();
        break;
      }

      block_chol_solve<BlockSize>(Gps, S.s, S.red_val);

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

      if (tid == 0) {
        const T zero_eps = (sizeof(T) == 4 ? T(1e-12) : T(1e-15));
        int new_n        = 0;
        for (int jj = 0; jj < np; ++jj) {
          int j_idx = S.idx[jj];
          if (S.x[j_idx] > zero_eps) {
            S.idx[new_n++] = j_idx;
          } else {
            S.act[j_idx] = 0;
            S.x[j_idx]   = T(0);
          }
        }
        sm_n_active = new_n;
      }
      __syncthreads();

      if (sm_n_active == 0) break;
    }

    if (sm_should_break_outer) break;
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

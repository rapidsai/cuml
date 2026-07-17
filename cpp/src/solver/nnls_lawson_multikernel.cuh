/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// -----------------------------------------------------------------------------
// Batched, step-synchronous Lawson-Hanson NNLS ("lawson_multikernel").
//
// Unlike nnls_lawson.cuh (which runs the whole active-set solve for one problem
// inside one persistent block, keeping G / Gp / state in shared memory), this
// backend advances the ENTIRE batch by one Lawson iteration per "global step".
// A step is a short, fixed pipeline of batched device operations over all P
// problems (grid = one block per problem for the custom kernels, batchCount = P
// for the cuBLAS / cuSOLVER primitives).  All per-problem state lives in GLOBAL
// memory, so there is no O(n^2) shared-memory cap and the design scales with n.
//
// Factor maintenance uses the batched linear-algebra primitives directly:
//   * grow  (add a column): incremental rank-1 Cholesky update, realised with
//     cublas<t>trsmBatched (solve L_11 y = g_new) + a device sqrt kernel for the
//     new diagonal L_22 = sqrt(G_jj - ||y||^2).  The host-syncing scalar sqrt
//     that forces a sync in raft::linalg::choleskyRank1Update stays on device
//     here, so the whole update is asynchronous.
//   * drop (remove columns): the active submatrix is re-factored with
//     cusolverDn<t>potrfBatched.  There is no batched Cholesky-downdate
//     primitive in cuBLAS / cuSOLVER, so a refactor of the (small) survivor set
//     is the primitive-based, sync-free realisation of the downdate.
//
// Because cuBLAS / cuSOLVER batched calls take a host-side batchCount and a
// single (uniform) per-matrix dimension, every problem's active submatrix is
// embedded in a fixed n x n SPD system: active columns occupy add-order slots
// [0, np) and the trailing slots [np, n) are identity padding with zero RHS, so
// the solve returns the active-set solution and zeros elsewhere.  The batched
// primitives therefore run over all P problems with dims (n, n) every step;
// per-problem participation is gated inside the custom kernels (a non-adder's
// grow RHS is zero, a non-dropper's refactor input is the identity).
//
// Stopping is non-synchronising: a device counter of finished problems is
// periodically copied (cudaMemcpyAsync) into a pinned host word that the host
// spins on.  The host may enqueue a few extra (no-op) steps due to the async
// lag; converged problems are inert, so this is harmless.
// -----------------------------------------------------------------------------

#include <cuml/common/pinned_host_vector.hpp>

#include <raft/core/cublas_macros.hpp>
#include <raft/core/cusolver_macros.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/resource/cublas_handle.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/cusolver_dn_handle.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cuda_rt_essentials.hpp>
#include <raft/util/cuda_utils.cuh>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cstdint>
#include <limits>
#include <optional>

namespace ML {
namespace Solver {
namespace detail {

// ---- Per-problem phase (state at the start of a step) -----------------------
// Only PH_ADD, PH_SOLVE and PH_DONE are ever observed at a step boundary;
// PH_GROWFIN and PH_REFAC are transient within a single step.
enum LawsonMkPhase : std::int8_t {
  PH_ADD     = 0,  ///< compute gradient, pick + activate the entering column
  PH_GROWFIN = 1,  ///< finish the rank-1 grow (write the new L row)
  PH_SOLVE   = 2,  ///< solve on the current active set, feasibility / ratio test
  PH_REFAC   = 3,  ///< re-factor the survivor submatrix after a drop
  PH_DONE    = 4   ///< optimal (or budget exhausted); inert
};

// ============================ batched primitive shims ========================
// Thin, direct cuBLAS / cuSOLVER batched wrappers (not yet exposed by raft),
// following the direct-call style used in lars_impl.cuh.

template <typename T>
inline cublasStatus_t mk_trsmBatched(cublasHandle_t handle,
                                     cublasFillMode_t uplo,
                                     cublasOperation_t trans,
                                     int m,
                                     int nrhs,
                                     const T* alpha,
                                     const T* const A[],
                                     int lda,
                                     T* const B[],
                                     int ldb,
                                     int batch);

template <>
inline cublasStatus_t mk_trsmBatched<float>(cublasHandle_t handle,
                                            cublasFillMode_t uplo,
                                            cublasOperation_t trans,
                                            int m,
                                            int nrhs,
                                            const float* alpha,
                                            const float* const A[],
                                            int lda,
                                            float* const B[],
                                            int ldb,
                                            int batch)
{
  return cublasStrsmBatched(handle, CUBLAS_SIDE_LEFT, uplo, trans, CUBLAS_DIAG_NON_UNIT, m, nrhs,
                            alpha, A, lda, B, ldb, batch);
}

template <>
inline cublasStatus_t mk_trsmBatched<double>(cublasHandle_t handle,
                                             cublasFillMode_t uplo,
                                             cublasOperation_t trans,
                                             int m,
                                             int nrhs,
                                             const double* alpha,
                                             const double* const A[],
                                             int lda,
                                             double* const B[],
                                             int ldb,
                                             int batch)
{
  return cublasDtrsmBatched(handle, CUBLAS_SIDE_LEFT, uplo, trans, CUBLAS_DIAG_NON_UNIT, m, nrhs,
                            alpha, A, lda, B, ldb, batch);
}

template <typename T>
inline cusolverStatus_t mk_potrfBatched(
  cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, T* Aarray[], int lda, int* info, int batch);

template <>
inline cusolverStatus_t mk_potrfBatched<float>(
  cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, float* Aarray[], int lda, int* info, int batch)
{
  return cusolverDnSpotrfBatched(handle, uplo, n, Aarray, lda, info, batch);
}

template <>
inline cusolverStatus_t mk_potrfBatched<double>(
  cusolverDnHandle_t handle, cublasFillMode_t uplo, int n, double* Aarray[], int lda, int* info, int batch)
{
  return cusolverDnDpotrfBatched(handle, uplo, n, Aarray, lda, info, batch);
}

template <typename T>
inline cublasStatus_t mk_gemm_nn(cublasHandle_t handle,
                                 int m,
                                 int n,
                                 int k,
                                 const T* alpha,
                                 const T* A,
                                 int lda,
                                 const T* B,
                                 int ldb,
                                 const T* beta,
                                 T* C,
                                 int ldc);

template <>
inline cublasStatus_t mk_gemm_nn<float>(cublasHandle_t handle,
                                        int m,
                                        int n,
                                        int k,
                                        const float* alpha,
                                        const float* A,
                                        int lda,
                                        const float* B,
                                        int ldb,
                                        const float* beta,
                                        float* C,
                                        int ldc)
{
  return cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
inline cublasStatus_t mk_gemm_nn<double>(cublasHandle_t handle,
                                         int m,
                                         int n,
                                         int k,
                                         const double* alpha,
                                         const double* A,
                                         int lda,
                                         const double* B,
                                         int ldb,
                                         const double* beta,
                                         double* C,
                                         int ldc)
{
  return cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

// =============================== device helpers ==============================

/** Small positive floor used to keep the Cholesky factor well defined. */
template <typename T>
__device__ __forceinline__ T mk_ridge()
{
  return sizeof(T) == 4 ? T(1e-7) : T(1e-14);
}

/** Threshold below which an updated coordinate is treated as having left the
 *  active set (matches nnls_lawson.cuh). */
template <typename T>
__device__ __forceinline__ T mk_zero_eps()
{
  return sizeof(T) == 4 ? T(1e-12) : T(1e-15);
}

/** Warp (single-block-warp) sum reduction. */
template <typename T>
__device__ __forceinline__ T mk_warp_sum(T v)
{
  for (int off = raft::WarpSize / 2; off > 0; off >>= 1)
    v += __shfl_xor_sync(0xffffffff, v, off);
  return v;
}

/** Warp min reduction. */
template <typename T>
__device__ __forceinline__ T mk_warp_min(T v)
{
  for (int off = raft::WarpSize / 2; off > 0; off >>= 1) {
    T o = __shfl_xor_sync(0xffffffff, v, off);
    if (o < v) v = o;
  }
  return v;
}

// ================================= kernels ===================================
// All custom kernels use grid.x = P and blockDim.x = raft::WarpSize (one warp
// per problem); n is small for this backend so a single warp with strided loops
// is sufficient and keeps the reductions in registers.

template <typename T>
__global__ void mk_init_kernel(
  T* L, T* X, int* idx, std::int8_t* act, int* nact, std::int8_t* phase, int* iters, int n, int P)
{
  const int p = blockIdx.x;
  if (p >= P) return;
  const int t  = threadIdx.x;
  const int nt = blockDim.x;
  T* Lp        = L + static_cast<std::size_t>(p) * n * n;
  for (int e = t; e < n * n; e += nt) {
    const int r = e % n;
    const int c = e / n;
    Lp[e]       = (r == c) ? T(1) : T(0);
  }
  for (int j = t; j < n; j += nt) {
    X[static_cast<std::size_t>(p) * n + j]   = T(0);
    act[static_cast<std::size_t>(p) * n + j] = 0;
    idx[static_cast<std::size_t>(p) * n + j] = 0;
  }
  if (t == 0) {
    nact[p]  = 0;
    phase[p] = PH_ADD;
    iters[p] = 0;
  }
}

/** Fill every problem's scratch matrix with the identity (base state before a
 *  potential refactor). */
template <typename T>
__global__ void mk_set_identity_kernel(T* M, int n, int P)
{
  const int p = blockIdx.x;
  if (p >= P) return;
  T* Mp = M + static_cast<std::size_t>(p) * n * n;
  for (int e = threadIdx.x; e < n * n; e += blockDim.x) {
    const int r = e % n;
    const int c = e / n;
    Mp[e]       = (r == c) ? T(1) : T(0);
  }
}

/** Build the constant pointer arrays consumed by the batched primitives. */
template <typename T>
__global__ void mk_build_ptrs_kernel(
  T* L, T* M, T* v1, T* v2, T** pL, T** pM, T** pv1, T** pv2, int n, int P)
{
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= P) return;
  pL[p]  = L + static_cast<std::size_t>(p) * n * n;
  pM[p]  = M + static_cast<std::size_t>(p) * n * n;
  pv1[p] = v1 + static_cast<std::size_t>(p) * n;
  pv2[p] = v2 + static_cast<std::size_t>(p) * n;
}

/** Phase PH_ADD: pick the entering column (argmax of the projected gradient
 *  over inactive, masked-in columns) and either finish (PH_DONE) or prepare the
 *  rank-1 grow RHS g_new in v1 (PH_GROWFIN). */
template <typename T>
__global__ void mk_argmax_activate_kernel(const T* W,
                                          const T* G,
                                          const std::uint8_t* masks,
                                          const std::int8_t* act,
                                          const int* idx,
                                          const int* nact,
                                          std::int8_t* phase,
                                          int* iters,
                                          int* newcol,
                                          T* v1,
                                          int n,
                                          int P,
                                          T tol,
                                          int max_iter)
{
  const int p = blockIdx.x;
  if (p >= P || phase[p] != PH_ADD) return;
  const int lane        = threadIdx.x;
  const bool has_mask   = masks != nullptr;
  const std::size_t off = static_cast<std::size_t>(p) * n;

  T best_v   = -std::numeric_limits<T>::infinity();
  int best_j = -1;
  for (int j = lane; j < n; j += raft::WarpSize) {
    if (act[off + j] == 0 && (!has_mask || masks[off + j] != 0)) {
      const T v = W[off + j];
      if (v > best_v || (v == best_v && j > best_j)) {
        best_v = v;
        best_j = j;
      }
    }
  }
  for (int o = raft::WarpSize / 2; o > 0; o >>= 1) {
    const T v   = __shfl_xor_sync(0xffffffff, best_v, o);
    const int j = __shfl_xor_sync(0xffffffff, best_j, o);
    if (v > best_v || (v == best_v && j > best_j)) {
      best_v = v;
      best_j = j;
    }
  }
  best_j = __shfl_sync(0xffffffff, best_j, 0);
  best_v = __shfl_sync(0xffffffff, best_v, 0);

  const int np = nact[p];
  if (best_j < 0 || best_v <= tol || np >= n || iters[p] >= max_iter) {
    if (lane == 0) phase[p] = PH_DONE;
    return;
  }

  // g_new[a] = G(idx[a], best_j) for a < np; zero padding above.
  for (int a = lane; a < n; a += raft::WarpSize) {
    T g = T(0);
    if (a < np) g = G[static_cast<std::size_t>(best_j) * n + idx[off + a]];
    v1[off + a] = g;
  }
  if (lane == 0) {
    newcol[p] = best_j;
    iters[p]  = iters[p] + 1;
    phase[p]  = PH_GROWFIN;
  }
}

/** Phase PH_GROWFIN: v1 now holds y = L_11^{-1} g_new (from trsm); write the new
 *  row of L and its diagonal, activate the column, advance to PH_SOLVE. */
template <typename T>
__global__ void mk_grow_finalize_kernel(T* L,
                                         const T* G,
                                         const T* v1,
                                         const int* newcol,
                                         int* idx,
                                         std::int8_t* act,
                                         int* nact,
                                         std::int8_t* phase,
                                         int n,
                                         int P)
{
  const int p = blockIdx.x;
  if (p >= P || phase[p] != PH_GROWFIN) return;
  const int lane        = threadIdx.x;
  const std::size_t off = static_cast<std::size_t>(p) * n;
  const int slot        = nact[p];
  const int jstar       = newcol[p];
  T* Lp                 = L + static_cast<std::size_t>(p) * n * n;

  T ssq = T(0);
  for (int a = lane; a < slot; a += raft::WarpSize) {
    const T y = v1[off + a];
    ssq += y * y;
  }
  ssq = mk_warp_sum(ssq);

  for (int a = lane; a < slot; a += raft::WarpSize)
    Lp[static_cast<std::size_t>(a) * n + slot] = v1[off + a];

  if (lane == 0) {
    const T gjj  = G[static_cast<std::size_t>(jstar) * n + jstar];
    T val        = gjj - ssq;
    const T floor = gjj * mk_ridge<T>() + mk_ridge<T>();
    if (!(val > floor)) val = floor;
    Lp[static_cast<std::size_t>(slot) * n + slot] = std::sqrt(val);
    idx[off + slot]                               = jstar;
    act[off + jstar]                              = 1;
    nact[p]                                       = slot + 1;
    phase[p]                                      = PH_SOLVE;
  }
}

/** Phase PH_SOLVE: build the RHS d' = c[idx[a]] (a < np), zero padding. */
template <typename T>
__global__ void mk_build_rhs_kernel(
  const T* C, const int* idx, const int* nact, const std::int8_t* phase, T* v2, int n, int P)
{
  const int p = blockIdx.x;
  if (p >= P || phase[p] != PH_SOLVE) return;
  const int lane        = threadIdx.x;
  const std::size_t off = static_cast<std::size_t>(p) * n;
  const int np          = nact[p];
  for (int a = lane; a < n; a += raft::WarpSize) {
    T d = T(0);
    if (a < np) d = C[off + idx[off + a]];
    v2[off + a] = d;
  }
}

/** Phase PH_SOLVE: v2 now holds s (the unconstrained active-set solution).
 *  Feasibility test -> commit; otherwise ratio-test partial step, drop the
 *  binding coordinates, and stage the survivor Gram in M for a refactor. */
template <typename T>
__global__ void mk_feas_ratio_drop_kernel(T* X,
                                          const T* G,
                                          const T* C,
                                          T* v2,
                                          int* idx,
                                          std::int8_t* act,
                                          int* nact,
                                          std::int8_t* phase,
                                          T* M,
                                          int n,
                                          int P)
{
  const int p = blockIdx.x;
  if (p >= P || phase[p] != PH_SOLVE) return;
  const int lane        = threadIdx.x;
  const std::size_t off = static_cast<std::size_t>(p) * n;
  int np                = nact[p];
  const T eps           = mk_zero_eps<T>();

  // min(s) over the active block
  T smin = std::numeric_limits<T>::infinity();
  for (int a = lane; a < np; a += raft::WarpSize) {
    const T s = v2[off + a];
    if (s < smin) smin = s;
  }
  smin = mk_warp_min(smin);

  if (smin > -eps) {
    // Feasible (up to a tiny tolerance): commit, clamping tiny negatives to 0.
    for (int a = lane; a < np; a += raft::WarpSize) {
      const T s             = v2[off + a];
      X[off + idx[off + a]] = s > T(0) ? s : T(0);
    }
    if (lane == 0) phase[p] = PH_ADD;
    return;
  }

  // Infeasible: alpha = min over binding coords of x / (x - s).  With smin <=
  // -eps there is a strictly negative s_a, so denom = x_a - s_a > 0 and alpha is
  // finite; the binding coordinate reaches ~0 and is dropped -> guaranteed
  // progress.
  T alpha = std::numeric_limits<T>::infinity();
  for (int a = lane; a < np; a += raft::WarpSize) {
    const T s = v2[off + a];
    if (s <= T(0)) {
      const T x     = X[off + idx[off + a]];
      const T denom = x - s;
      if (denom > T(0)) {
        const T r = x / denom;
        if (r < alpha) alpha = r;
      }
    }
  }
  alpha = mk_warp_min(alpha);
  if (!(alpha < std::numeric_limits<T>::infinity())) alpha = T(0);

  // Partial step on the active set.
  for (int a = lane; a < np; a += raft::WarpSize) {
    const int j = idx[off + a];
    const T x   = X[off + j];
    X[off + j]  = x + alpha * (v2[off + a] - x);
  }
  __syncwarp();

  // Compact survivors (x > eps) on lane 0; drop the rest.
  int w = 0;
  if (lane == 0) {
    for (int a = 0; a < np; ++a) {
      const int j = idx[off + a];
      if (X[off + j] > eps) {
        idx[off + w++] = j;
      } else {
        act[off + j] = 0;
        X[off + j]   = T(0);
      }
    }
    nact[p]  = w;
    phase[p] = PH_REFAC;
  }
  // Publish lane 0's idx compaction / count to the rest of the warp.
  __threadfence_block();
  __syncwarp();
  np = __shfl_sync(0xffffffff, w, 0);

  // Stage the embedded survivor Gram (identity padding) for potrfBatched.
  T* Mp = M + static_cast<std::size_t>(p) * n * n;
  for (int e = lane; e < n * n; e += raft::WarpSize) {
    const int r = e % n;
    const int c = e / n;
    T v;
    if (r < np && c < np) {
      v = G[static_cast<std::size_t>(idx[off + c]) * n + idx[off + r]];
      if (r == c) v += v * mk_ridge<T>() + mk_ridge<T>();  // tiny ridge for potrf
    } else {
      v = (r == c) ? T(1) : T(0);
    }
    Mp[e] = v;
  }
}

/** Phase PH_REFAC: M holds the freshly factored L (from potrfBatched); copy it
 *  into the persistent factor and go back to PH_SOLVE. */
template <typename T>
__global__ void mk_refac_finalize_kernel(
  T* L, const T* M, const std::int8_t* phase, int n, int P)
{
  const int p = blockIdx.x;
  if (p >= P || phase[p] != PH_REFAC) return;
  T* Lp       = L + static_cast<std::size_t>(p) * n * n;
  const T* Mp = M + static_cast<std::size_t>(p) * n * n;
  for (int e = threadIdx.x; e < n * n; e += blockDim.x)
    Lp[e] = Mp[e];
}

/** Flip PH_REFAC -> PH_SOLVE after the copy (separate pass so the copy is not
 *  racing the phase read). */
__global__ void mk_refac_advance_kernel(std::int8_t* phase, int P)
{
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= P) return;
  if (phase[p] == PH_REFAC) phase[p] = PH_SOLVE;
}

/** Count problems that have reached PH_DONE. */
__global__ void mk_count_done_kernel(const std::int8_t* phase, int* n_done, int P)
{
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= P) return;
  if (phase[p] == PH_DONE) atomicAdd(n_done, 1);
}

// ================================ host driver ================================

template <typename T>
inline void nnls_lawson_multikernel_dispatch(
  raft::resources const&                                             handle,
  raft::device_matrix_view<const T, int, raft::col_major>            G,
  raft::device_matrix_view<const T, int, raft::col_major>            C,
  std::optional<raft::device_matrix_view<const std::uint8_t, int, raft::col_major>> masks,
  raft::device_matrix_view<T, int, raft::col_major>                  X,
  int                                                                max_iter,
  T                                                                  tol)
{
  const int n = G.extent(0);
  const int P = X.extent(1);
  if (P <= 0 || n <= 0) return;
  if (max_iter <= 0) max_iter = 3 * n + 1;

  cudaStream_t      stream   = raft::resource::get_cuda_stream(handle);
  cublasHandle_t    cublas_h = raft::resource::get_cublas_handle(handle);
  cusolverDnHandle_t cusolver_h = raft::resource::get_cusolver_dn_handle(handle);
  RAFT_CUBLAS_TRY(cublasSetStream(cublas_h, stream));
  RAFT_CUSOLVER_TRY(cusolverDnSetStream(cusolver_h, stream));

  const std::size_t nn = static_cast<std::size_t>(n) * n;
  const std::size_t nP = static_cast<std::size_t>(n) * P;

  // ---- state -------------------------------------------------------------
  rmm::device_uvector<T> L(nn * P, stream);       // per-problem Cholesky factor
  rmm::device_uvector<T> M(nn * P, stream);       // refactor scratch
  rmm::device_uvector<T> W(nP, stream);           // projected gradient
  rmm::device_uvector<T> v1(nP, stream);          // grow RHS / y
  rmm::device_uvector<T> v2(nP, stream);          // solve RHS / s
  rmm::device_uvector<int> idx(nP, stream);
  rmm::device_uvector<std::int8_t> act(nP, stream);
  rmm::device_uvector<int> nact(P, stream);
  rmm::device_uvector<std::int8_t> phase(P, stream);
  rmm::device_uvector<int> iters(P, stream);
  rmm::device_uvector<int> newcol(P, stream);
  rmm::device_uvector<int> info(P, stream);
  rmm::device_scalar<int> n_done(stream);

  rmm::device_uvector<T*> pL(P, stream);
  rmm::device_uvector<T*> pM(P, stream);
  rmm::device_uvector<T*> pv1(P, stream);
  rmm::device_uvector<T*> pv2(P, stream);

  const T* Gp = G.data_handle();
  const T* Cp = C.data_handle();
  T*       Xp = X.data_handle();
  const std::uint8_t* Mkp = masks.has_value() ? masks->data_handle() : nullptr;

  constexpr int WARP  = raft::WarpSize;
  const int lin_block = 256;
  const int lin_grid  = (P + lin_block - 1) / lin_block;

  mk_init_kernel<T><<<P, WARP, 0, stream>>>(
    L.data(), Xp, idx.data(), act.data(), nact.data(), phase.data(), iters.data(), n, P);
  mk_build_ptrs_kernel<T><<<lin_grid, lin_block, 0, stream>>>(
    L.data(), M.data(), v1.data(), v2.data(), pL.data(), pM.data(), pv1.data(), pv2.data(), n, P);

  const T one = T(1);
  const T neg = T(-1);

  // pinned convergence mirror; polled without a stream sync
  ML::pinned_host_vector<int> done_host(1);
  done_host[0] = 0;

  const int poll_every    = 8;
  const long long max_step = static_cast<long long>(max_iter) * (n + 2) + 32;

  for (long long step = 0; step < max_step; ++step) {
    // 1. gradient  W = C - G X
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      W.data(), Cp, nP * sizeof(T), cudaMemcpyDeviceToDevice, stream));
    RAFT_CUBLAS_TRY(mk_gemm_nn<T>(
      cublas_h, n, P, n, &neg, Gp, n, Xp, n, &one, W.data(), n));

    // 2. pick + activate entering column (PH_ADD -> PH_GROWFIN / PH_DONE)
    mk_argmax_activate_kernel<T><<<P, WARP, 0, stream>>>(
      W.data(), Gp, Mkp, act.data(), idx.data(), nact.data(), phase.data(), iters.data(),
      newcol.data(), v1.data(), n, P, tol, max_iter);

    // 3. rank-1 grow: solve L y = g_new  (forward)
    RAFT_CUBLAS_TRY(mk_trsmBatched<T>(
      cublas_h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, 1, &one,
      const_cast<const T* const*>(pL.data()), n, pv1.data(), n, P));

    // 4. write the new L row / diagonal (PH_GROWFIN -> PH_SOLVE)
    mk_grow_finalize_kernel<T><<<P, WARP, 0, stream>>>(
      L.data(), Gp, v1.data(), newcol.data(), idx.data(), act.data(), nact.data(), phase.data(),
      n, P);

    // 5. build the solve RHS d' (PH_SOLVE)
    mk_build_rhs_kernel<T><<<P, WARP, 0, stream>>>(
      Cp, idx.data(), nact.data(), phase.data(), v2.data(), n, P);

    // 6. solve L L^T s = d'  (forward then back)
    RAFT_CUBLAS_TRY(mk_trsmBatched<T>(
      cublas_h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, n, 1, &one,
      const_cast<const T* const*>(pL.data()), n, pv2.data(), n, P));
    RAFT_CUBLAS_TRY(mk_trsmBatched<T>(
      cublas_h, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, n, 1, &one,
      const_cast<const T* const*>(pL.data()), n, pv2.data(), n, P));

    // 7. reset refactor scratch to identity (non-droppers stay identity)
    mk_set_identity_kernel<T><<<P, WARP, 0, stream>>>(M.data(), n, P);

    // 8. feasibility / ratio test / drop (PH_SOLVE -> PH_ADD or PH_REFAC)
    mk_feas_ratio_drop_kernel<T><<<P, WARP, 0, stream>>>(
      Xp, Gp, Cp, v2.data(), idx.data(), act.data(), nact.data(), phase.data(), M.data(), n, P);

    // 9. refactor survivor submatrix (PH_REFAC), then copy back and advance
    RAFT_CUSOLVER_TRY(mk_potrfBatched<T>(
      cusolver_h, CUBLAS_FILL_MODE_LOWER, n, pM.data(), n, info.data(), P));
    mk_refac_finalize_kernel<T><<<P, WARP, 0, stream>>>(
      L.data(), M.data(), phase.data(), n, P);
    mk_refac_advance_kernel<<<lin_grid, lin_block, 0, stream>>>(phase.data(), P);

    // 10. non-synchronising convergence check
    if ((step % poll_every) == 0) {
      RAFT_CUDA_TRY(cudaMemsetAsync(n_done.data(), 0, sizeof(int), stream));
      mk_count_done_kernel<<<lin_grid, lin_block, 0, stream>>>(phase.data(), n_done.data(), P);
      RAFT_CUDA_TRY(cudaMemcpyAsync(
        done_host.data(), n_done.data(), sizeof(int), cudaMemcpyDeviceToHost, stream));
      // Read the value published by an EARLIER async copy (no stream sync).
      const int seen = *static_cast<volatile int*>(static_cast<void*>(done_host.data()));
      if (seen >= P) break;
    }
  }
  // The caller synchronises the stream (nnls_batched_impl / the Cython layer);
  // X already holds the committed non-negative solutions.
}

}  // namespace detail
}  // namespace Solver
}  // namespace ML

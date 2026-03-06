/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "edge_metric.cuh"

#include <cuml/common/utils.hpp>
#include <cuml/manifold/umapparams.h>

#include <raft/core/operators.hpp>
#include <raft/linalg/map_reduce.cuh>
#include <raft/sparse/coo.hpp>
#include <raft/stats/meanvar.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <cmath>

namespace UMAPAlgo {
namespace DensMap {

using namespace ML;

template <typename T>
struct DensMapData {
  rmm::device_uvector<T> R_storage;
  rmm::device_uvector<T> mu_storage;
  rmm::device_uvector<T> re_sum_storage;
  rmm::device_uvector<T> phi_sum_storage;
  rmm::device_uvector<T> exp_neg_re_sum_storage;

  T* R;
  T* mu;
  T* re_sum;
  T* phi_sum;
  T* exp_neg_re_sum;

  T mu_tot;
  T re_std;
  T re_mean;
  T re_cov;
  T dens_lambda;

  T cov_over_var;
  T outer_scale;

  int n_vertices;

  DensMapData(int n_verts, int nnz, T lambda, cudaStream_t stream)
    : R_storage(n_verts, stream),
      mu_storage(nnz, stream),
      re_sum_storage(n_verts, stream),
      phi_sum_storage(n_verts, stream),
      exp_neg_re_sum_storage(n_verts, stream),
      R(R_storage.data()),
      mu(mu_storage.data()),
      re_sum(re_sum_storage.data()),
      phi_sum(phi_sum_storage.data()),
      exp_neg_re_sum(exp_neg_re_sum_storage.data()),
      mu_tot(0),
      re_std(0),
      re_mean(0),
      re_cov(0),
      dens_lambda(lambda),
      cov_over_var(0),
      outer_scale(0),
      n_vertices(n_verts)
  {
  }
};

/**
 * Scatter-add over edges: ro[v] += mu[i] * D^2[i], mu_sum[v] += mu[i]
 * for both endpoints j=rows[i], k=cols[i].
 */
template <typename T, int TPB_X = 256>
CUML_KERNEL void compute_ro_scatter_kernel(
  const int* rows, const int* cols, const T* mu, const T* edge_dists, T* ro, T* mu_sum, int nnz)
{
  int idx = blockIdx.x * TPB_X + threadIdx.x;
  if (idx >= nnz) return;

  int j  = rows[idx];
  int k  = cols[idx];
  T mu_i = mu[idx];
  T D_sq = edge_dists[idx];
  T mu_D = mu_i * D_sq;

  atomicAdd(&ro[j], mu_D);
  atomicAdd(&ro[k], mu_D);
  atomicAdd(&mu_sum[j], mu_i);
  atomicAdd(&mu_sum[k], mu_i);
}

/**
 * Normalize ro: ro[v] = log(eps + ro[v] / mu_sum[v])
 */
template <typename T, int TPB_X = 256>
CUML_KERNEL void normalize_ro_kernel(T* ro, const T* mu_sum, int n)
{
  int idx = blockIdx.x * TPB_X + threadIdx.x;
  if (idx >= n) return;

  T epsilon = T(1e-8);
  ro[idx]   = log(epsilon + (ro[idx] / mu_sum[idx]));
}

/**
 * Standardize in-place: data[v] = (data[v] - mean) / std
 */
template <typename T, int TPB_X = 256>
CUML_KERNEL void standardize_kernel(T* data, T mean, T std_val, int n)
{
  int idx = blockIdx.x * TPB_X + threadIdx.x;
  if (idx >= n) return;

  data[idx] = (data[idx] - mean) / std_val;
}

/**
 * Precompute densMAP original-space density data from the (trimmed) COO graph
 * and the raw input data X.
 *
 * Returns a filled DensMapData with R, mu, mu_tot ready for the
 * optimization loop.
 */
template <typename T, typename nnz_t, int TPB_X = 256>
std::unique_ptr<DensMapData<T>> densmap_precompute(const T* X,
                                                   int n_vertices,
                                                   int n_features,
                                                   raft::sparse::COO<T>& graph,
                                                   UMAPParams* params,
                                                   cudaStream_t stream)
{
  nnz_t nnz = graph.nnz;
  auto dm   = std::make_unique<DensMapData<T>>(n_vertices, nnz, params->dens_lambda, stream);

  raft::copy(dm->mu, graph.vals(), nnz, stream);

  // 1. Compute per-edge distances from X using the configured metric
  rmm::device_uvector<T> edge_dists(nnz, stream);
  EdgeMetric::launch_edge_dists_kernel<T, TPB_X>(X,
                                                 graph.rows(),
                                                 graph.cols(),
                                                 edge_dists.data(),
                                                 nnz,
                                                 n_features,
                                                 params->metric,
                                                 static_cast<T>(params->p),
                                                 stream);

  // 2. Compute ro and mu_sum via scatter-add (mu_sum is only needed locally)
  rmm::device_uvector<T> mu_sum(n_vertices, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(dm->R, 0, n_vertices * sizeof(T), stream));  // use R storage as ro
  RAFT_CUDA_TRY(cudaMemsetAsync(mu_sum.data(), 0, n_vertices * sizeof(T), stream));
  {
    dim3 grid(raft::ceildiv(nnz, static_cast<nnz_t>(TPB_X)), 1, 1);
    dim3 blk(TPB_X, 1, 1);
    compute_ro_scatter_kernel<T, TPB_X><<<grid, blk, 0, stream>>>(
      graph.rows(), graph.cols(), dm->mu, edge_dists.data(), dm->R, mu_sum.data(), nnz);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  // 3. Normalize ro = log(eps + ro / mu_sum)
  {
    dim3 grid(raft::ceildiv(n_vertices, TPB_X), 1, 1);
    dim3 blk(TPB_X, 1, 1);
    normalize_ro_kernel<T, TPB_X><<<grid, blk, 0, stream>>>(dm->R, mu_sum.data(), n_vertices);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  // 4. Compute mean and std of ro (stored in R), then standardize R = (ro - mean) / std
  {
    rmm::device_uvector<T> d_mean(1, stream);
    rmm::device_uvector<T> d_var(1, stream);
    raft::stats::meanvar(d_mean.data(), d_var.data(), dm->R, 1, n_vertices, false, true, stream);

    T ro_mean, ro_var;
    raft::update_host(&ro_mean, d_mean.data(), 1, stream);
    raft::update_host(&ro_var, d_var.data(), 1, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

    T ro_std = std::sqrt(static_cast<float>(ro_var));
    if (ro_std < T(1e-10)) ro_std = T(1e-10);

    dim3 grid(raft::ceildiv(n_vertices, TPB_X), 1, 1);
    dim3 blk(TPB_X, 1, 1);
    standardize_kernel<T, TPB_X><<<grid, blk, 0, stream>>>(dm->R, ro_mean, ro_std, n_vertices);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }

  // 5. Compute mu_tot = sum(mu_sum) / 2
  {
    rmm::device_uvector<T> d_total(1, stream);
    raft::linalg::mapReduce(d_total.data(),
                            static_cast<size_t>(n_vertices),
                            T(0.0),
                            raft::identity_op{},
                            raft::add_op{},
                            stream,
                            mu_sum.data());
    T total;
    raft::update_host(&total, d_total.data(), 1, stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    dm->mu_tot = total / T(2.0);
  }

  return dm;
}

// ---- Per-epoch embedding-space density computation ----

/**
 * Per-edge scatter-add: accumulate phi*d^2 into re_sum and phi into phi_sum
 * for both endpoints.
 */
template <typename T, int TPB_X = 256>
CUML_KERNEL void densmap_epoch_init_scatter_kernel(const T* __restrict__ head_embedding,
                                                   const T* __restrict__ tail_embedding,
                                                   const int* __restrict__ head,
                                                   const int* __restrict__ tail,
                                                   int nnz,
                                                   int n_components,
                                                   T a,
                                                   T b,
                                                   T* __restrict__ re_sum,
                                                   T* __restrict__ phi_sum)
{
  int idx = blockIdx.x * TPB_X + threadIdx.x;
  if (idx >= nnz) return;

  int j = head[idx];
  int k = tail[idx];

  const size_t j_off = static_cast<size_t>(j) * n_components;
  const size_t k_off = static_cast<size_t>(k) * n_components;

  T dist_sq;
  if (n_components == 2) {
    T d0    = head_embedding[j_off] - tail_embedding[k_off];
    T d1    = head_embedding[j_off + 1] - tail_embedding[k_off + 1];
    dist_sq = d0 * d0 + d1 * d1;
  } else if (n_components == 3) {
    T d0    = head_embedding[j_off] - tail_embedding[k_off];
    T d1    = head_embedding[j_off + 1] - tail_embedding[k_off + 1];
    T d2    = head_embedding[j_off + 2] - tail_embedding[k_off + 2];
    dist_sq = d0 * d0 + d1 * d1 + d2 * d2;
  } else {
    dist_sq = T(0.0);
    for (int d = 0; d < n_components; d++) {
      T diff = head_embedding[j_off + d] - tail_embedding[k_off + d];
      dist_sq += diff * diff;
    }
  }

  T phi      = T(1.0) / (T(1.0) + a * pow(dist_sq, b));
  T phi_dist = phi * dist_sq;

  atomicAdd(&re_sum[j], phi_dist);
  atomicAdd(&re_sum[k], phi_dist);
  atomicAdd(&phi_sum[j], phi);
  atomicAdd(&phi_sum[k], phi);
}

/**
 * Normalize re_sum: re_sum[v] = log(eps + re_sum[v] / phi_sum[v])
 * and precompute exp_neg[v] = exp(-re_sum[v]) to avoid per-edge exp() calls.
 */
template <typename T, int TPB_X = 256>
CUML_KERNEL void densmap_normalize_re_kernel(T* re_sum, const T* phi_sum, T* exp_neg_re_sum, int n)
{
  int idx = blockIdx.x * TPB_X + threadIdx.x;
  if (idx >= n) return;

  T epsilon           = T(1e-8);
  T val               = log(epsilon + (re_sum[idx] / phi_sum[idx]));
  re_sum[idx]         = val;
  exp_neg_re_sum[idx] = exp(-val);
}

/**
 * Run the per-epoch densMAP init: compute re_sum and phi_sum from the
 * current embedding.
 */
template <typename T, typename nnz_t, int TPB_X = 256>
void densmap_epoch_init(T* head_embedding,
                        T* tail_embedding,
                        const int* head,
                        const int* tail,
                        nnz_t nnz,
                        int n_vertices,
                        int n_components,
                        T a,
                        T b,
                        T* re_sum,
                        T* phi_sum,
                        T* exp_neg_re_sum,
                        cudaStream_t stream)
{
  RAFT_CUDA_TRY(cudaMemsetAsync(re_sum, 0, n_vertices * sizeof(T), stream));
  RAFT_CUDA_TRY(cudaMemsetAsync(phi_sum, 0, n_vertices * sizeof(T), stream));

  dim3 grid(raft::ceildiv(nnz, static_cast<nnz_t>(TPB_X)), 1, 1);
  dim3 blk(TPB_X, 1, 1);
  densmap_epoch_init_scatter_kernel<T, TPB_X><<<grid, blk, 0, stream>>>(
    head_embedding, tail_embedding, head, tail, nnz, n_components, a, b, re_sum, phi_sum);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  dim3 grid_n(raft::ceildiv(n_vertices, TPB_X), 1, 1);
  densmap_normalize_re_kernel<T, TPB_X>
    <<<grid_n, blk, 0, stream>>>(re_sum, phi_sum, exp_neg_re_sum, n_vertices);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

/**
 * Compute per-epoch statistics: re_std, re_mean, re_cov.
 * Writes results back into the DensMapData struct on the host.
 */
template <typename T>
void densmap_epoch_stats(DensMapData<T>& dm, float dens_var_shift, cudaStream_t stream)
{
  int n = dm.n_vertices;

  rmm::device_uvector<T> d_mean(1, stream);
  rmm::device_uvector<T> d_var(1, stream);
  rmm::device_uvector<T> d_dot(1, stream);

  raft::stats::meanvar(d_mean.data(), d_var.data(), dm.re_sum, 1, n, false, true, stream);

  raft::linalg::mapReduce(d_dot.data(),
                          static_cast<size_t>(n),
                          T(0.0),
                          raft::mul_op{},
                          raft::add_op{},
                          stream,
                          dm.re_sum,
                          dm.R);

  T h_mean, h_var, h_dot;
  raft::update_host(&h_mean, d_mean.data(), 1, stream);
  raft::update_host(&h_var, d_var.data(), 1, stream);
  raft::update_host(&h_dot, d_dot.data(), 1, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  dm.re_mean = h_mean;
  dm.re_std  = std::sqrt(static_cast<float>(h_var) + dens_var_shift);
  dm.re_cov  = h_dot / T(n - 1);

  T re_std_sq     = dm.re_std * dm.re_std;
  dm.cov_over_var = dm.re_cov / re_std_sq;
  dm.outer_scale  = dm.dens_lambda * dm.mu_tot / (dm.re_std * T(n));
}

// ---- Device helper for densMAP gradient coefficient ----

template <typename T>
DI T compute_densmap_grad_coeff(T dist_squared,
                                T a,
                                T b,
                                int j,
                                int k,
                                T mu_edge,
                                T re_mean,
                                T cov_over_var,
                                T outer_scale,
                                const T* __restrict__ R,
                                const T* __restrict__ re_sum,
                                const T* __restrict__ phi_sum,
                                const T* __restrict__ exp_neg_re_sum)
{
  T pow_db    = pow(dist_squared, b);
  T denom     = T(1.0) + a * pow_db;
  T phi       = T(1.0) / denom;
  T dphi_term = a * b * pow_db * phi / dist_squared;

  T q_jk = phi / phi_sum[k];
  T q_kj = phi / phi_sum[j];

  T b_phi_term = T(1.0) - b * (T(1.0) - phi);
  T drk        = q_jk * (b_phi_term * exp_neg_re_sum[k] + dphi_term);
  T drj        = q_kj * (b_phi_term * exp_neg_re_sum[j] + dphi_term);

  T weight_k = R[k] - cov_over_var * (re_sum[k] - re_mean);
  T weight_j = R[j] - cov_over_var * (re_sum[j] - re_mean);

  return outer_scale * (weight_k * drk + weight_j * drj) / mu_edge;
}

}  // namespace DensMap
}  // namespace UMAPAlgo

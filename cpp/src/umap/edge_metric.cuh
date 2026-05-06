/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuml/common/distance_type.hpp>
#include <cuml/common/utils.hpp>

#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

namespace UMAPAlgo {
namespace EdgeMetric {

template <typename T>
DI T warp_reduce_sum(T val)
{
  for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  return val;
}

template <typename T>
DI T warp_reduce_max(T val)
{
  for (int offset = 16; offset > 0; offset >>= 1)
    val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
  return val;
}

/** Per-feature accumulation — compiled once per metric, zero branches in the hot loop. */
template <ML::distance::DistanceType metric, typename T>
DI void edge_dist_accumulate(T vj, T vk, T* acc, T p_val)
{
  using DT = ML::distance::DistanceType;
  if constexpr (metric == DT::L2Expanded || metric == DT::L2Unexpanded ||
                metric == DT::L2SqrtExpanded || metric == DT::L2SqrtUnexpanded) {
    T d = vj - vk;
    acc[0] += d * d;
  } else if constexpr (metric == DT::L1) {
    T d = vj - vk;
    acc[0] += (d >= T(0)) ? d : -d;
  } else if constexpr (metric == DT::Linf) {
    T d    = vj - vk;
    T a    = (d >= T(0)) ? d : -d;
    acc[0] = (a > acc[0]) ? a : acc[0];
  } else if constexpr (metric == DT::LpUnexpanded) {
    T d = vj - vk;
    acc[0] += pow((d >= T(0)) ? d : -d, p_val);
  } else if constexpr (metric == DT::CosineExpanded) {
    acc[0] += vj * vk;
    acc[1] += vj * vj;
    acc[2] += vk * vk;
  } else if constexpr (metric == DT::InnerProduct) {
    acc[0] += vj * vk;
  } else if constexpr (metric == DT::CorrelationExpanded) {
    acc[0] += vj * vk;
    acc[1] += vj;
    acc[2] += vk;
    acc[3] += vj * vj;
    acc[4] += vk * vk;
  } else if constexpr (metric == DT::Canberra) {
    T d   = vj - vk;
    T ad  = (d >= T(0)) ? d : -d;
    T den = ((vj >= T(0)) ? vj : -vj) + ((vk >= T(0)) ? vk : -vk);
    if (den > T(0)) acc[0] += ad / den;
  } else if constexpr (metric == DT::HellingerExpanded) {
    T d = sqrt((vj >= T(0)) ? vj : T(0)) - sqrt((vk >= T(0)) ? vk : T(0));
    acc[0] += d * d;
  } else if constexpr (metric == DT::BrayCurtis) {
    T d = vj - vk;
    acc[0] += (d >= T(0)) ? d : -d;
    T s = vj + vk;
    acc[1] += (s >= T(0)) ? s : -s;
  } else if constexpr (metric == DT::JensenShannon) {
    T m = (vj + vk) * T(0.5);
    if (m > T(0)) {
      if (vj > T(0)) acc[0] += vj * log(vj / m);
      if (vk > T(0)) acc[0] += vk * log(vk / m);
    }
  } else if constexpr (metric == DT::KLDivergence) {
    if (vj > T(0) && vk > T(0)) acc[0] += vj * log(vj / vk);
  } else if constexpr (metric == DT::HammingUnexpanded) {
    acc[0] += (vj != vk) ? T(1) : T(0);
  } else if constexpr (metric == DT::JaccardExpanded) {
    acc[0] += min(vj, vk);
    acc[1] += max(vj, vk);
  } else if constexpr (metric == DT::RusselRaoExpanded) {
    acc[0] += (vj != T(0) && vk != T(0)) ? T(1) : T(0);
  } else if constexpr (metric == DT::DiceExpanded) {
    bool a = vj != T(0), b = vk != T(0);
    if (a && b) acc[0] += T(1);
    if (a || b) acc[1] += T(1);
  } else {
    T d = vj - vk;
    acc[0] += d * d;
  }
}

/** Convert the reduced accumulators into a scalar distance. */
template <ML::distance::DistanceType metric, typename T>
DI T edge_dist_finalize(const T* acc, int n_features, T p_val)
{
  using DT = ML::distance::DistanceType;
  if constexpr (metric == DT::L2Expanded || metric == DT::L2Unexpanded || metric == DT::L1 ||
                metric == DT::Linf || metric == DT::Canberra || metric == DT::KLDivergence)
    return acc[0];
  else if constexpr (metric == DT::L2SqrtExpanded || metric == DT::L2SqrtUnexpanded ||
                     metric == DT::HellingerExpanded)
    return sqrt(acc[0]);
  else if constexpr (metric == DT::LpUnexpanded)
    return pow(acc[0], T(1) / p_val);
  else if constexpr (metric == DT::JensenShannon)
    return sqrt(acc[0] * T(0.5));
  else if constexpr (metric == DT::InnerProduct)
    return T(1) - acc[0];
  else if constexpr (metric == DT::HammingUnexpanded)
    return acc[0] / T(n_features);
  else if constexpr (metric == DT::RusselRaoExpanded)
    return T(1) - acc[0] / T(n_features);
  else if constexpr (metric == DT::CosineExpanded) {
    T d = sqrt(acc[1] * acc[2]);
    return (d > T(1e-10)) ? T(1) - acc[0] / d : T(0);
  } else if constexpr (metric == DT::CorrelationExpanded) {
    T n = T(n_features), mj = acc[1] / n, mk = acc[2] / n;
    T d = sqrt((acc[3] / n - mj * mj) * (acc[4] / n - mk * mk));
    return (d > T(1e-10)) ? T(1) - (acc[0] / n - mj * mk) / d : T(0);
  } else if constexpr (metric == DT::BrayCurtis)
    return (acc[1] > T(0)) ? acc[0] / acc[1] : T(0);
  else if constexpr (metric == DT::JaccardExpanded)
    return (acc[1] > T(0)) ? T(1) - acc[0] / acc[1] : T(0);
  else if constexpr (metric == DT::DiceExpanded) {
    T d = acc[0] + acc[1];
    return (d > T(0)) ? T(1) - T(2) * acc[0] / d : T(0);
  } else
    return sqrt(acc[0]);
}

/**
 * Compute pairwise distance for each edge (j, k) in the COO graph.
 *
 * One warp (32 threads) cooperates on each edge: lanes split the feature
 * dimension, giving coalesced reads and a fast warp-shuffle reduction.
 * Templated on the metric so the inner loop is branch-free.
 */
template <ML::distance::DistanceType metric, typename T, int TPB_X = 256>
CUML_KERNEL void compute_edge_dists_kernel(
  const T* X, const int* rows, const int* cols, T* edge_dists, int nnz, int n_features, T p_val)
{
  using DT                             = ML::distance::DistanceType;
  static constexpr int WARP_SIZE       = 32;
  static constexpr int WARPS_PER_BLOCK = TPB_X / WARP_SIZE;

  int lane     = threadIdx.x & (WARP_SIZE - 1);
  int edge_idx = blockIdx.x * WARPS_PER_BLOCK + (threadIdx.x / WARP_SIZE);
  if (edge_idx >= nnz) return;

  const T* xj = X + static_cast<size_t>(rows[edge_idx]) * n_features;
  const T* xk = X + static_cast<size_t>(cols[edge_idx]) * n_features;

  if constexpr (metric == DT::Haversine) {
    if (lane == 0) {
      T dlat = xk[0] - xj[0], dlon = xk[1] - xj[1];
      T sl = sin(dlat * T(0.5)), sn = sin(dlon * T(0.5));
      edge_dists[edge_idx] = T(2) * asin(sqrt(sl * sl + cos(xj[0]) * cos(xk[0]) * sn * sn));
    }
  } else {
    T acc[5] = {T(0), T(0), T(0), T(0), T(0)};
    for (int d = lane; d < n_features; d += WARP_SIZE)
      edge_dist_accumulate<metric>(xj[d], xk[d], acc, p_val);

    if constexpr (metric == DT::Linf)
      acc[0] = warp_reduce_max(acc[0]);
    else
      acc[0] = warp_reduce_sum(acc[0]);
    for (int i = 1; i < 5; i++)
      acc[i] = warp_reduce_sum(acc[i]);

    if (lane == 0) edge_dists[edge_idx] = edge_dist_finalize<metric>(acc, n_features, p_val);
  }
}

/** Runtime metric dispatch — single switch at launch, not in the hot loop. */
template <typename T, int TPB_X = 256>
void launch_edge_dists_kernel(const T* X,
                              const int* rows,
                              const int* cols,
                              T* edge_dists,
                              int nnz,
                              int n_features,
                              ML::distance::DistanceType metric,
                              T p_val,
                              cudaStream_t stream)
{
  using DT                             = ML::distance::DistanceType;
  static constexpr int WARPS_PER_BLOCK = TPB_X / 32;
  dim3 grid(raft::ceildiv(nnz, WARPS_PER_BLOCK));
  dim3 blk(TPB_X);

  // clang-format off
#define EDGE_METRIC_CASE(M)                                                   \
  case DT::M:                                                                 \
    compute_edge_dists_kernel<DT::M, T, TPB_X><<<grid, blk, 0, stream>>>(    \
      X, rows, cols, edge_dists, nnz, n_features, p_val);                     \
    break

  switch (metric) {
    EDGE_METRIC_CASE(L2Expanded);
    EDGE_METRIC_CASE(L2SqrtExpanded);
    EDGE_METRIC_CASE(L2Unexpanded);
    EDGE_METRIC_CASE(L2SqrtUnexpanded);
    EDGE_METRIC_CASE(CosineExpanded);
    EDGE_METRIC_CASE(L1);
    EDGE_METRIC_CASE(InnerProduct);
    EDGE_METRIC_CASE(Linf);
    EDGE_METRIC_CASE(Canberra);
    EDGE_METRIC_CASE(LpUnexpanded);
    EDGE_METRIC_CASE(CorrelationExpanded);
    EDGE_METRIC_CASE(JaccardExpanded);
    EDGE_METRIC_CASE(HellingerExpanded);
    EDGE_METRIC_CASE(Haversine);
    EDGE_METRIC_CASE(BrayCurtis);
    EDGE_METRIC_CASE(JensenShannon);
    EDGE_METRIC_CASE(HammingUnexpanded);
    EDGE_METRIC_CASE(KLDivergence);
    EDGE_METRIC_CASE(RusselRaoExpanded);
    EDGE_METRIC_CASE(DiceExpanded);
    default:
      compute_edge_dists_kernel<DT::L2SqrtExpanded, T, TPB_X><<<grid, blk, 0, stream>>>(
        X, rows, cols, edge_dists, nnz, n_features, p_val);
      break;
  }
#undef EDGE_METRIC_CASE
  // clang-format on

  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace EdgeMetric
}  // namespace UMAPAlgo

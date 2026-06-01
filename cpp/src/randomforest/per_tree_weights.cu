/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "per_tree_weights.cuh"

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda_runtime.h>

namespace ML {

namespace {

// Shared-mem atomicAdd bincount; one block, smem holds n_unique_labels
// ints. Simpler than CUB/Thrust at RFC's typical n_unique_labels (<= ~64).
template <typename LabelT>
__global__ void bincount_bootstrap_kernel(const LabelT* labels,
                                          const int* selected_rows,
                                          int n_sampled_rows,
                                          int n_unique_labels,
                                          int* counts_out)
{
  extern __shared__ int shared_counts[];
  for (int c = threadIdx.x; c < n_unique_labels; c += blockDim.x) {
    shared_counts[c] = 0;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < n_sampled_rows; i += blockDim.x) {
    int row    = selected_rows[i];
    LabelT lab = labels[row];
    atomicAdd(&shared_counts[static_cast<int>(lab)], 1);
  }
  __syncthreads();

  for (int c = threadIdx.x; c < n_unique_labels; c += blockDim.x) {
    counts_out[c] = shared_counts[c];
  }
}

template <typename DataT, typename LabelT>
__global__ void fill_per_tree_weights_kernel(const LabelT* labels,
                                             int n_rows,
                                             const int* counts,
                                             int n_sampled_rows,
                                             const int* n_classes_present_ptr,
                                             const DataT* base_sample_weight,
                                             DataT* per_tree_weights)
{
  // sklearn parity: compute_sample_weight uses len(classes_subsample) as
  // the denominator, not full class cardinality. n_classes_present_ptr
  // is a device int written by count_classes_present_kernel; reading it
  // here avoids a host-side cudaStreamSynchronize per tree.
  const double denom_const = static_cast<double>(*n_classes_present_ptr);
  const double n_sampled   = static_cast<double>(n_sampled_rows);

  int tid    = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = tid; i < n_rows; i += stride) {
    int lab   = static_cast<int>(labels[i]);
    int count = counts[lab];
    double reciprocal =
      (count > 0) ? (n_sampled / (denom_const * static_cast<double>(count))) : 0.0;
    double base         = base_sample_weight ? static_cast<double>(base_sample_weight[i]) : 1.0;
    per_tree_weights[i] = static_cast<DataT>(base * reciprocal);
  }
}

// Tiny reduction: count how many class bins received at least one sample
// from this bootstrap. Used as the per-tree class-cardinality denominator
// in fill_per_tree_weights_kernel to match sklearn.
__global__ void count_classes_present_kernel(const int* counts,
                                             int n_unique_labels,
                                             int* n_present_out)
{
  extern __shared__ int shared_n_present[];
  if (threadIdx.x == 0) shared_n_present[0] = 0;
  __syncthreads();

  for (int c = threadIdx.x; c < n_unique_labels; c += blockDim.x) {
    if (counts[c] > 0) atomicAdd(&shared_n_present[0], 1);
  }
  __syncthreads();
  if (threadIdx.x == 0) *n_present_out = shared_n_present[0];
}

}  // namespace

template <typename DataT, typename LabelT>
void computePerTreeBalancedWeights(const raft::handle_t& handle,
                                   cudaStream_t stream,
                                   const LabelT* labels,
                                   const int* selected_rows,
                                   int n_sampled_rows,
                                   int n_unique_labels,
                                   int n_rows,
                                   const DataT* base_sample_weight,
                                   const double* /* class_weight_array */,
                                   DataT* per_tree_weights)
{
  // Per-tree compute is bounded by stream s; rmm-deferred-free preserves
  // lifetime past DecisionTree::fit's async kernels on the same stream.
  // Shared-memory budget bound: n_unique_labels <= 8192 fits the typical
  // 48-100KB per-block smem limit at sizeof(int) per class.
  ASSERT(static_cast<size_t>(n_unique_labels) <= 8192,
         "computePerTreeBalancedWeights: n_unique_labels=%d exceeds the "
         "8192 cap dictated by shared-memory budget on the bincount kernel",
         n_unique_labels);

  rmm::device_uvector<int> counts(n_unique_labels, stream);
  rmm::device_uvector<int> n_classes_present(1, stream);

  constexpr int TPB   = 256;
  size_t shared_bytes = sizeof(int) * static_cast<size_t>(n_unique_labels);
  bincount_bootstrap_kernel<LabelT><<<1, TPB, shared_bytes, stream>>>(
    labels, selected_rows, n_sampled_rows, n_unique_labels, counts.data());

  count_classes_present_kernel<<<1, TPB, sizeof(int), stream>>>(
    counts.data(), n_unique_labels, n_classes_present.data());

  // Three kernel launches on `stream`, all async. No host sync needed:
  // fill_per_tree_weights_kernel dereferences n_classes_present on device.
  int fill_blocks = (n_rows + TPB - 1) / TPB;
  fill_per_tree_weights_kernel<DataT, LabelT>
    <<<fill_blocks, TPB, 0, stream>>>(labels,
                                      n_rows,
                                      counts.data(),
                                      n_sampled_rows,
                                      n_classes_present.data(),
                                      base_sample_weight,
                                      per_tree_weights);
}

// Explicit template instantiations matching the (T, L) combinations cuml's
// RandomForest::fit supports (see randomforest.cu). The LabelT=float regressor
// path is kept for ABI symmetry; BALANCED_SUBSAMPLE is RFC-only.
template void computePerTreeBalancedWeights<float, int>(const raft::handle_t&,
                                                        cudaStream_t,
                                                        const int*,
                                                        const int*,
                                                        int,
                                                        int,
                                                        int,
                                                        const float*,
                                                        const double*,
                                                        float*);
template void computePerTreeBalancedWeights<double, int>(const raft::handle_t&,
                                                         cudaStream_t,
                                                         const int*,
                                                         const int*,
                                                         int,
                                                         int,
                                                         int,
                                                         const double*,
                                                         const double*,
                                                         double*);

}  // namespace ML

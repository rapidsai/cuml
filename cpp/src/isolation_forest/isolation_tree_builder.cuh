/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Lightweight batched Isolation Forest tree builder.
 * 
 * Key design decisions:
 * - All trees built in a single kernel launch (1 block per tree)
 * - Subsamples gathered into contiguous buffer to avoid scattered reads
 * - Tree built iteratively using a stack (no recursion in CUDA)
 * - Random splits: pick random feature, then random threshold in [min, max]
 */

#pragma once

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <thrust/transform.h>

#include <cmath>

namespace ML {
namespace IsolationTree {

constexpr int MAX_DEPTH = 16;  // Supports up to 2^16 = 65536 samples per tree
constexpr int MAX_NODES = (1 << (MAX_DEPTH + 1)) - 1;  // Complete binary tree nodes

/**
 * @brief Tree node structure.
 * Internal nodes: feature_idx >= 0, threshold is split value
 * Leaf nodes: feature_idx = -1, threshold stores sample count (for c(n) adjustment)
 */
struct IFNode {
  int feature_idx;
  float threshold;
  int left_child;
  int right_child;
};

template <typename T>
struct IFTree {
  IFNode nodes[MAX_NODES];
  int n_nodes;
  int max_depth;
};

/** @brief Check if pointer is on GPU (for input validation). */
inline bool is_dev_ptr(const void* ptr)
{
  cudaPointerAttributes attr;
  auto err = cudaPointerGetAttributes(&attr, ptr);
  if (err != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged;
}

/**
 * @brief Build one isolation tree from pre-gathered subsample data.
 * 
 * Uses iterative DFS with explicit stack 
 * Only thread 0 builds the tree; other threads help with initialization.
 * 
 * @param local_data Subsample data in row-major [max_samples x n_cols]
 */
template <typename T>
__device__ void build_tree_iterative_local(
    const T* __restrict__ local_data,
    int n_samples,
    int n_cols,
    int max_samples,
    int max_depth,
    curandState* rng_state,
    IFTree<T>* tree)
{
  struct StackEntry { int node_idx, start_idx, end_idx, depth; };
  
  // Shared memory layout: [stack for DFS] [work_indices for partitioning]
  extern __shared__ char shared_mem[];
  StackEntry* stack = reinterpret_cast<StackEntry*>(shared_mem);
  int* work_indices = reinterpret_cast<int*>(shared_mem + sizeof(int) * 4 * MAX_DEPTH * 2);
  
  // Initialize work_indices = [0, 1, 2, ..., n_samples-1]
  int tid = threadIdx.x;
  for (int i = tid; i < n_samples; i += blockDim.x) {
    work_indices[i] = i;
  }
  __syncthreads();
  
  if (tid == 0) {
    tree->n_nodes = 0;
    tree->max_depth = max_depth;
  }
  __syncthreads();
  
  // Single-threaded tree construction (tree building is inherently sequential)
  if (tid == 0) {
    int stack_top = 0;
    stack[stack_top++] = {0, 0, n_samples, 0};
    tree->n_nodes = 1;
    
    while (stack_top > 0) {
      StackEntry entry = stack[--stack_top];
      int node_idx = entry.node_idx;
      int start = entry.start_idx;
      int end = entry.end_idx;
      int depth = entry.depth;
      int n_node_samples = end - start;
      
      // Stopping condition: max depth or isolated sample
      if (depth >= max_depth || n_node_samples <= 1) {
        tree->nodes[node_idx] = {-1, static_cast<float>(n_node_samples), -1, -1};
        continue;
      }
      
      int feature = curand(rng_state) % n_cols;
      
      // Find min/max of selected feature in current partition
      T min_val = local_data[work_indices[start] * n_cols + feature];
      T max_val = min_val;
      for (int i = start + 1; i < end; i++) {
        T val = local_data[work_indices[i] * n_cols + feature];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
      }
      
      // All values identical -> can't split, make leaf
      if (min_val >= max_val) {
        tree->nodes[node_idx] = {-1, static_cast<float>(n_node_samples), -1, -1};
        continue;
      }
      
      // Random threshold in (min, max) - core of Isolation Forest
      float rand_frac = curand_uniform(rng_state);
      T threshold = min_val + static_cast<T>(rand_frac) * (max_val - min_val);
      
      // Partition: move samples < threshold to left side (in-place swap)
      int left_end = start;
      for (int i = start; i < end; i++) {
        T val = local_data[work_indices[i] * n_cols + feature];
        if (val < threshold) {
          int tmp = work_indices[left_end];
          work_indices[left_end] = work_indices[i];
          work_indices[i] = tmp;
          left_end++;
        }
      }
      
      // Edge case: all samples went to one side (shouldn't happen)
      if (left_end == start || left_end == end) {
        left_end = start + n_node_samples / 2;
      }
      
      int left_child = tree->n_nodes;
      int right_child = tree->n_nodes + 1;
      tree->n_nodes += 2;
      
      tree->nodes[node_idx] = {feature, static_cast<float>(threshold), left_child, right_child};
      
      // Push children (right first so left is processed first - DFS order)
      stack[stack_top++] = {right_child, left_end, end, depth + 1};
      stack[stack_top++] = {left_child, start, left_end, depth + 1};
    }
  }
  __syncthreads();
}

/**
 * @brief Main kernel: each block builds one complete isolation tree.
 * 
 * Phase 1: Generate random sample indices and gather data into contiguous buffer
 * Phase 2: Build tree from gathered data (avoids scattered global memory reads)
 */
template <typename T>
__global__ void build_isolation_trees_kernel(
    const T* __restrict__ data,         // Full dataset [n_rows x n_cols] column-major
    int n_rows,
    int n_cols,
    int n_trees,
    int max_samples,
    int max_depth,
    uint64_t seed,
    IFTree<T>* trees,
    T* __restrict__ subsample_buffer)   // [n_trees x max_samples x n_cols] row-major
{
  int tree_id = blockIdx.x;
  if (tree_id >= n_trees) return;
  
  // Each tree gets unique RNG state (seed + tree_id)
  curandState rng_state;
  curand_init(seed, tree_id, 0, &rng_state);
  
  T* local_data = subsample_buffer + tree_id * max_samples * n_cols;
  
  extern __shared__ char shared_mem[];
  int* sample_indices = reinterpret_cast<int*>(shared_mem + sizeof(int) * 4 * MAX_DEPTH * 2);
  
  // Thread 0 generates random sample indices
  if (threadIdx.x == 0) {
    for (int i = 0; i < max_samples; i++) {
      sample_indices[i] = curand(&rng_state) % n_rows;
    }
  }
  __syncthreads();
  
  // All threads gather subsample data (coalesced writes to local_data).
  // We copy regardless of input layout to get 256 samples into a contiguous
  // buffer for cache-friendly tree building. The col-major→row-major conversion
  // is just different index math during this already-required copy.
  int tid = threadIdx.x;
  for (int s = 0; s < max_samples; s++) {
    int src_row = sample_indices[s];
    for (int f = tid; f < n_cols; f += blockDim.x) {
      local_data[s * n_cols + f] = data[src_row + f * n_rows];
    }
  }
  __syncthreads();
  
  build_tree_iterative_local(local_data, max_samples, n_cols, max_samples, max_depth, &rng_state, &trees[tree_id]);
}

/**
 * @brief Traverse tree to compute path length for one sample.
 * 
 * Path length = depth + c(n) adjustment for leaf's remaining samples.
 * c(n) accounts for expected additional depth if tree were fully grown.
 */
template <typename T>
__device__ T traverse_tree(const IFTree<T>* tree, const T* sample, int n_cols)
{
  int node_idx = 0;
  int depth = 0;
  
  while (true) {
    const IFNode& node = tree->nodes[node_idx];
    
    if (node.feature_idx < 0) {
      // Leaf: add c(n) for unbuilt subtree
      int n_samples = static_cast<int>(node.threshold);
      T c_n = T(0);
      if (n_samples > 1) {
        // c(n) = 2*H(n-1) - 2*(n-1)/n, H(k) ≈ ln(k) + γ
        T n = static_cast<T>(n_samples);
        c_n = T(2) * (log(n - T(1)) + T(0.5772156649)) - T(2) * (n - T(1)) / n;
      }
      return static_cast<T>(depth) + c_n;
    }
    
    T val = sample[node.feature_idx];
    node_idx = (val < static_cast<T>(node.threshold)) ? node.left_child : node.right_child;
    depth++;
  }
}

/**
 * @brief Compute average path length across all trees for each sample.
 * Each thread handles one sample.
 */
template <typename T>
__global__ void compute_path_lengths_kernel(
    const T* __restrict__ data,     // [n_samples x n_cols] row-major
    int n_samples,
    int n_cols,
    const IFTree<T>* __restrict__ trees,
    int n_trees,
    T* __restrict__ path_lengths)   // Output: [n_samples]
{
  int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample_idx >= n_samples) return;
  
  const T* sample = data + sample_idx * n_cols;
  
  T total_path = T(0);
  for (int t = 0; t < n_trees; t++) {
    total_path += traverse_tree(&trees[t], sample, n_cols);
  }
  
  path_lengths[sample_idx] = total_path / static_cast<T>(n_trees);
}

/**
 * @brief Host function to build all isolation trees.
 * 
 * Allocates temporary buffer for gathered subsamples, launches kernel,
 * then buffer is freed automatically when function returns.
 */
template <typename T>
void build_isolation_forest(
    const raft::handle_t& handle,
    const T* data,              // [n_rows x n_cols] column-major
    int n_rows,
    int n_cols,
    int n_trees,
    int max_samples,
    int max_depth,
    uint64_t seed,
    IFTree<T>* d_trees)         // Output: device array [n_trees]
{
  auto stream = handle.get_stream();
  
  // Temp buffer: [n_trees x max_samples x n_cols] for gathered subsamples
  // ~10MB for typical 100 trees x 256 samples x 100 features x float
  size_t subsample_buffer_size = static_cast<size_t>(n_trees) * max_samples * n_cols;
  rmm::device_uvector<T> subsample_buffer(subsample_buffer_size, stream);
  
  // Shared memory: stack for DFS + sample indices
  int stack_size = MAX_DEPTH * 2 * 4 * sizeof(int);
  int indices_size = max_samples * sizeof(int);
  int shared_mem_size = stack_size + indices_size;
  
  build_isolation_trees_kernel<T><<<n_trees, 128, shared_mem_size, stream>>>(
      data, n_rows, n_cols, n_trees, max_samples, max_depth, seed, d_trees, subsample_buffer.data());
  
  RAFT_CUDA_TRY(cudaGetLastError());
}

}  // namespace IsolationTree
}  // namespace ML

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
 * - Leaf nodes store pre-computed path lengths (depth + c(n)) so that
 *   inference is a simple tree traversal with no per-leaf computation
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
 * @brief Compute c(n) = 2H(n-1) - 2(n-1)/n, the expected path length in an
 * unsuccessful BST search (used for adjusting isolation depth at leaves).
 */
__device__ __forceinline__ float compute_c_n(int n_samples)
{
  if (n_samples <= 1) return 0.0f;
  if (n_samples == 2) return 1.0f;
  float n = static_cast<float>(n_samples);
  return 2.0f * (logf(n - 1.0f) + 0.5772156649f) - 2.0f * (n - 1.0f) / n;
}

/**
 * @brief Tree node structure.
 * Internal nodes: feature_idx >= 0, threshold is split value
 * Leaf nodes: feature_idx = -1, threshold stores pre-computed path length
 *             (depth + c(n_samples)) for direct use by inference
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
        float path_length = static_cast<float>(depth) + compute_c_n(n_node_samples);
        tree->nodes[node_idx] = {-1, path_length, -1, -1};
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
        float path_length = static_cast<float>(depth) + compute_c_n(n_node_samples);
        tree->nodes[node_idx] = {-1, path_length, -1, -1};
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
    size_t n_rows,                       // Use size_t to support large datasets (>2B elements)
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
  size_t* sample_indices = reinterpret_cast<size_t*>(shared_mem + sizeof(int) * 4 * MAX_DEPTH * 2);
  
  // Thread 0 generates random sample indices
  if (threadIdx.x == 0) {
    for (int i = 0; i < max_samples; i++) {
      // Use 64-bit random for proper sampling from large datasets (>2B rows)
      uint64_t rand64 = (static_cast<uint64_t>(curand(&rng_state)) << 32) | curand(&rng_state);
      sample_indices[i] = rand64 % n_rows;
    }
  }
  __syncthreads();
  
  // All threads gather subsample data (coalesced writes to local_data).
  // We copy regardless of input layout to get 256 samples into a contiguous
  // buffer for cache-friendly tree building. The col-major→row-major conversion
  // is just different index math during this already-required copy.
  int tid = threadIdx.x;
  for (int s = 0; s < max_samples; s++) {
    size_t src_row = sample_indices[s];
    for (int f = tid; f < n_cols; f += blockDim.x) {
      local_data[s * n_cols + f] = data[src_row + static_cast<size_t>(f) * n_rows];
    }
  }
  __syncthreads();
  
  build_tree_iterative_local(local_data, max_samples, n_cols, max_samples, max_depth, &rng_state, &trees[tree_id]);
}

/**
 * @brief Traverse tree to get path length for one sample.
 *
 * Leaf nodes store pre-computed path lengths (depth + c(n)), so traversal
 * simply walks to the leaf and returns the stored value.
 */
template <typename T>
__device__ T traverse_tree(const IFTree<T>* tree, const T* sample, int n_cols)
{
  int node_idx = 0;
  
  while (true) {
    const IFNode& node = tree->nodes[node_idx];
    
    if (node.feature_idx < 0) {
      return static_cast<T>(node.threshold);
    }
    
    T val = sample[node.feature_idx];
    node_idx = (val < static_cast<T>(node.threshold)) ? node.left_child : node.right_child;
  }
}

/**
 * @brief Compute average path length across all trees for each sample.
 * Each thread handles one sample.
 */
template <typename T>
__global__ void compute_path_lengths_kernel(
    const T* __restrict__ data,     // [n_samples x n_cols] row-major
    size_t n_samples,
    int n_cols,
    const IFTree<T>* __restrict__ trees,
    int n_trees,
    T* __restrict__ path_lengths)   // Output: [n_samples]
{
  size_t sample_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (sample_idx >= n_samples) return;
  
  const T* sample = data + sample_idx * n_cols;
  
  T total_path = T(0);
  for (int t = 0; t < n_trees; t++) {
    total_path += traverse_tree(&trees[t], sample, n_cols);
  }
  
  path_lengths[sample_idx] = (n_trees > 0) ? total_path / static_cast<T>(n_trees) : T(0);
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
    size_t n_rows,              // Use size_t to support large datasets
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
  
  // Shared memory: stack for DFS + sample indices (size_t for >2B row support)
  int stack_size = MAX_DEPTH * 2 * 4 * sizeof(int);
  int indices_size = max_samples * sizeof(size_t);
  int shared_mem_size = stack_size + indices_size;
  
  build_isolation_trees_kernel<T><<<n_trees, 128, shared_mem_size, stream>>>(
      data, n_rows, n_cols, n_trees, max_samples, max_depth, seed, d_trees, subsample_buffer.data());
  
  RAFT_CUDA_TRY(cudaGetLastError());
}

// ── Tree compaction: extract only used nodes for efficient D2H transfer ──────

/**
 * @brief Extract per-tree metadata (n_nodes, max_depth) from the padded
 * IFTree structs. One thread per tree.
 */
template <typename T>
__global__ void extract_tree_metadata_kernel(
    const IFTree<T>* __restrict__ trees,
    int n_trees,
    int* __restrict__ n_nodes_out,
    int* __restrict__ max_depth_out)
{
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= n_trees) return;
  n_nodes_out[t]  = trees[t].n_nodes;
  max_depth_out[t] = trees[t].max_depth;
}

/**
 * @brief Copy each tree's used nodes into a contiguous output buffer.
 * One block per tree, threads cooperate on the copy.
 */
template <typename T>
__global__ void compact_trees_kernel(
    const IFTree<T>* __restrict__ trees,
    int n_trees,
    const int* __restrict__ offsets,
    IFNode* __restrict__ compact_nodes)
{
  int tree_id = blockIdx.x;
  if (tree_id >= n_trees) return;
  int n = trees[tree_id].n_nodes;
  int off = offsets[tree_id];
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    compact_nodes[off + i] = trees[tree_id].nodes[i];
  }
}

/**
 * @brief Compact an isolation forest from padded IFTree structs into
 * host-side vectors, minimizing the device-to-host transfer.
 *
 * Typical transfer: ~400 KB (100 trees × 255 nodes × 16B) instead of
 * ~200 MB (100 trees × 131K nodes × 16B).
 *
 * @param[in]  handle          RAFT handle
 * @param[in]  d_trees         Device pointer to padded IFTree<T> array
 * @param[in]  n_trees         Number of trees
 * @param[out] h_nodes         All used nodes concatenated (host)
 * @param[out] h_tree_offsets  Start offset in h_nodes for each tree (host)
 * @param[out] h_tree_n_nodes  Node count per tree (host)
 * @param[out] h_tree_max_depth Max depth per tree (host)
 */
template <typename T>
void compact_isolation_forest(
    const raft::handle_t& handle,
    const IFTree<T>* d_trees,
    int n_trees,
    std::vector<IFNode>& h_nodes,
    std::vector<int>& h_tree_offsets,
    std::vector<int>& h_tree_n_nodes,
    std::vector<int>& h_tree_max_depth)
{
  auto stream = handle.get_stream();

  // 1. Extract metadata from padded structs
  rmm::device_uvector<int> d_n_nodes(n_trees, stream);
  rmm::device_uvector<int> d_max_depth(n_trees, stream);

  int threads = 128;
  int blocks  = (n_trees + threads - 1) / threads;
  extract_tree_metadata_kernel<T><<<blocks, threads, 0, stream>>>(
      d_trees, n_trees, d_n_nodes.data(), d_max_depth.data());
  RAFT_CUDA_TRY(cudaGetLastError());

  // 2. Copy metadata to host and compute prefix-sum offsets
  h_tree_n_nodes.resize(n_trees);
  h_tree_max_depth.resize(n_trees);
  RAFT_CUDA_TRY(cudaMemcpyAsync(h_tree_n_nodes.data(), d_n_nodes.data(),
                                  n_trees * sizeof(int), cudaMemcpyDeviceToHost, stream));
  RAFT_CUDA_TRY(cudaMemcpyAsync(h_tree_max_depth.data(), d_max_depth.data(),
                                  n_trees * sizeof(int), cudaMemcpyDeviceToHost, stream));
  handle.sync_stream(stream);

  h_tree_offsets.resize(n_trees);
  int total_nodes = 0;
  for (int t = 0; t < n_trees; ++t) {
    h_tree_offsets[t] = total_nodes;
    total_nodes += h_tree_n_nodes[t];
  }

  // 3. Compact nodes on device using the offsets
  rmm::device_uvector<int> d_offsets(n_trees, stream);
  RAFT_CUDA_TRY(cudaMemcpyAsync(d_offsets.data(), h_tree_offsets.data(),
                                  n_trees * sizeof(int), cudaMemcpyHostToDevice, stream));

  rmm::device_uvector<IFNode> d_compact(total_nodes, stream);
  compact_trees_kernel<T><<<n_trees, 128, 0, stream>>>(
      d_trees, n_trees, d_offsets.data(), d_compact.data());
  RAFT_CUDA_TRY(cudaGetLastError());

  // 4. Single compact D2H copy
  h_nodes.resize(total_nodes);
  RAFT_CUDA_TRY(cudaMemcpyAsync(h_nodes.data(), d_compact.data(),
                                  total_nodes * sizeof(IFNode), cudaMemcpyDeviceToHost, stream));
  handle.sync_stream(stream);
}

}  // namespace IsolationTree
}  // namespace ML

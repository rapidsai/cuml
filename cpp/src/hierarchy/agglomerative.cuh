/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuml/common/logger.hpp>

#include <cuml/cuml_api.h>
#include <raft/cudart_utils.h>
#include <common/cumlHandle.hpp>
#include <raft/cuda_utils.cuh>

#include <raft/mr/device/buffer.hpp>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace ML {
namespace Linkage {
namespace Label {
namespace Agglomerative {

template <typename value_idx, typename value_t>
class UnionFind {
 public:
  value_idx next_label;
  value_idx *parent;
  value_idx *size;

  value_idx n_indices;

  UnionFind(value_idx N_) {
    n_indices = 2 * N_ - 1;
    parent = new value_idx[n_indices];
    size = new value_idx[n_indices];

    next_label = N_;

    for (int i = 0; i < n_indices; i++) {
      parent[i] = -1;
      size[i] = i < N_ ? 1 : 0;
    }
  }

  value_idx find(value_idx n) {
    value_idx p;
    p = n;

    while (parent[n] != -1) n = parent[n];

    // path compression
    while (parent[p] != n) {
      value_idx ind = p == -1 ? n_indices - 1 : p;
      p = parent[ind];
      ind = p == -1 ? n_indices - 1 : p;
      parent[ind] = n;
    }
    return n;
  }

  void perform_union(value_idx m, value_idx n) {
    size[next_label] = size[m] + size[n];
    parent[m] = next_label;
    parent[n] = next_label;

    size[next_label] = size[m] + size[n];
    next_label += 1;
  }

  ~UnionFind() {
    delete[] parent;
    delete[] size;
  }
};

/**
 * Standard single-threaded agglomerative labeling on host. This should work
 * well for smaller sizes of m. This is a C++ port of the original reference
 * implementation of HDBSCAN.
 *
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle the raft handle
 * @param[in] rows src edges of the sorted MST
 * @param[in] cols dst edges of the sorted MST
 * @param[in] nnz the number of edges in the sorted MST
 * @param[out] out_src parents of output
 * @param[out] out_dst children of output
 * @param[out] out_delta distances of output
 * @param[out] out_size cluster sizes of output
 */
template <typename value_idx, typename value_t>
void build_dendrogram_host(const raft::handle_t &handle, const value_idx *rows,
                           const value_idx *cols, const value_t *data,
                           size_t nnz,
                           raft::mr::device::buffer<value_idx> &children,
                           raft::mr::device::buffer<value_t> &out_delta,
                           raft::mr::device::buffer<value_idx> &out_size) {
  auto d_alloc = handle.get_device_allocator();
  auto stream = handle.get_stream();

  value_idx n_edges = nnz;

  children.resize(n_edges * 2, stream);
  out_delta.resize(n_edges, stream);
  out_size.resize(n_edges, stream);

  std::vector<value_idx> mst_src_h(n_edges);
  std::vector<value_idx> mst_dst_h(n_edges);
  std::vector<value_t> mst_weights_h(n_edges);

  std::vector<value_idx> children_h(n_edges * 2);
  std::vector<value_t> out_delta_h(n_edges);
  std::vector<value_idx> out_size_h(n_edges);

  raft::update_host(mst_src_h.data(), rows, n_edges, stream);
  raft::update_host(mst_dst_h.data(), cols, n_edges, stream);
  raft::update_host(mst_weights_h.data(), data, n_edges, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  value_idx a, aa, b, bb;
  value_t delta;

  UnionFind<value_idx, value_t> U(nnz + 1);

  for (int i = 0; i < nnz; i++) {
    a = mst_src_h.data()[i];
    b = mst_dst_h.data()[i];

    delta = mst_weights_h.data()[i];

    aa = U.find(a);
    bb = U.find(b);

    int children_idx = i * 2;

    children_h[children_idx] = aa;
    children_h[children_idx + 1] = bb;
    out_delta_h[i] = delta;
    out_size_h[i] = U.size[aa] + U.size[bb];

    U.perform_union(aa, bb);
  }

  raft::update_device(children.data(), children_h.data(), n_edges * 2, stream);
  raft::update_device(out_delta.data(), out_delta_h.data(), n_edges, stream);
  raft::update_device(out_size.data(), out_size_h.data(), n_edges, stream);
}

/**
 * Parallel agglomerative labeling. This amounts to a parallel Kruskal's
 * MST algorithm, which breaks apart the sorted MST results into overlapping
 * subsets and independently runs Kruskal's algorithm on each subset,
 * merging them back together into a single hierarchy when complete.
 *
 * This outputs the same format as the reference HDBSCAN, but as 4 separate
 * arrays, rather than a single 2D array.
 *
 * Reference: http://cucis.ece.northwestern.edu/publications/pdf/HenPat12.pdf
 *
 * TODO: Investigate potential for the following end-to-end single-hierarchy batching:
 *    For each of k (independent) batches over the input:
 *    - Sample n elements from X
 *    - Compute mutual reachability graph of batch
 *    - Construct labels from batch
 *
 * The sampled datasets should have some overlap across batches. This will
 * allow for the cluster hierarchies to be merged. Being able to batch
 * will reduce the memory cost so that the full n^2 pairwise distances
 * don't need to be materialized in memory all at once.
 *
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle the raft handle
 * @param[in] rows src edges of the sorted MST
 * @param[in] cols dst edges of the sorted MST
 * @param[in] nnz the number of edges in the sorted MST
 * @param[out] out_src parents of output
 * @param[out] out_dst children of output
 * @param[out] out_delta distances of output
 * @param[out] out_size cluster sizes of output
 * @param[in] k_folds number of folds for parallelizing label step
 */
template <typename value_idx, typename value_t>
void build_dendrogram_device(const raft::handle_t &handle,
                             const value_idx *rows, const value_idx *cols,
                             const value_t *data, value_idx nnz,
                             value_idx *children, value_t *out_delta,
                             value_idx *out_size, value_idx k_folds) {
  ASSERT(k_folds < nnz / 2, "k_folds must be < n_edges / 2");
  /**
   * divide (sorted) mst coo into overlapping subsets. Easiest way to do this is to
   * break it into k-folds and iterate through two folds at a time.
   */

  // 1. Generate ranges for the overlapping subsets

  // 2. Run union-find in parallel for each pair of folds

  // 3. Sort individual label hierarchies

  // 4. Merge label hierarchies together
}

template <typename value_idx>
__global__ void write_levels_kernel(const value_idx *children,
                                    value_idx *parents, size_t n_leaves) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid < (n_leaves - 1) * 2) {
    value_idx level = tid / 2;
    value_idx child = children[tid];
    parents[child] = level;
  }
}

/**
 * Instead of propagating a label from roots to children,
 * the children each iterate up the tree until they find
 * the label of their parent. This increases the potential
 * parallelism.
 * @tparam value_idx
 * @param children
 * @param parents
 * @param n_leaves
 * @param labels
 */
template <typename value_idx>
__global__ void inherit_labels(const value_idx *children,
                               const value_idx *levels, size_t n_leaves,
                               value_idx *labels, int cut_level) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < ((n_leaves - 1) * 2) - 1) {
    value_idx node = children[tid];

    value_idx cur_level = tid / 2;

    /**
     * Any roots above the cut level should be ignored.
     * Any leaves at the cut level should already be labeled
     */
    if (cur_level > cut_level) return;

    value_idx cur_parent = node;

    value_idx label = labels[cur_parent];

    while (label == -1) {
      cur_parent = cur_level + n_leaves;

      cur_level = levels[cur_parent];

      label = labels[cur_parent];
    }

    printf("tid=%d, node=%d, label=%d\n", tid, node, label);
    labels[node] = label;
  }
}

template <typename value_idx>
struct init_label_roots {
  init_label_roots(value_idx *labels_) : labels(labels_) {}

  template <typename Tuple>
  __host__ __device__ void operator()(Tuple t) {
    labels[thrust::get<1>(t)] = thrust::get<0>(t);
  }

 private:
  value_idx *labels;
};

/**
 * Cuts the dendrogram at a particular level where the number of nodes
 * is equal to n_clusters, then propagates the resulting labels
 * to all the children.
 *
 * @tparam value_idx
 * @param handle
 * @param labels
 * @param children
 * @param n_clusters
 * @param n_leaves
 */
template <typename value_idx>
void extract_flattened_clusters(
  const raft::handle_t &handle, value_idx *labels,
  const raft::mr::device::buffer<value_idx> &children, value_idx n_clusters,
  size_t n_leaves) {
  auto d_alloc = handle.get_device_allocator();
  auto stream = handle.get_stream();

  /**
   * Compute levels for each node
   *
   *     1. Initialize "levels" array of size n_leaves * 2
   *
   *     2. For each entry in children, write parent
   *        out for each of the children
   */
  raft::mr::device::buffer<value_idx> levels(handle.get_device_allocator(),
                                             stream, n_leaves * 2);

  int n_blocks = raft::ceildiv((n_leaves - 1) * 2, (size_t)1024);
  write_levels_kernel<<<n_blocks, 1024, 0, stream>>>(children.data(),
                                                     levels.data(), n_leaves);
  /**
   * Step 1: Find label roots:
   *
   *     1. Copying children[children.size()-(n_clusters-1):] entries to
   *        separate array
   *     2. sort array
   *     3. take first n_clusters entries
   */
  raft::mr::device::buffer<value_idx> label_roots(handle.get_device_allocator(),
                                                  stream, (n_clusters - 1) * 2);

  value_idx children_cpy_start = children.size() - ((n_clusters - 1) * 2);
  raft::copy_async(label_roots.data(), children.data() + children_cpy_start,
                   (n_clusters - 1) * 2, stream);

  thrust::device_ptr<value_idx> t_label_roots =
    thrust::device_pointer_cast(label_roots.data());

  thrust::sort(thrust::cuda::par.on(stream), t_label_roots,
               t_label_roots + ((n_clusters - 1) * 2),
               thrust::greater<value_idx>());

  raft::mr::device::buffer<value_idx> tmp_labels(handle.get_device_allocator(),
                                                 stream, (n_leaves - 1) * 2);

  // Init labels to -1
  thrust::device_ptr<value_idx> t_labels =
    thrust::device_pointer_cast(tmp_labels.data());
  thrust::fill(t_labels, t_labels + ((n_leaves - 1) * 2), -1);

  // Write labels for cluster roots to "labels"
  thrust::counting_iterator<uint> first(0);
  thrust::counting_iterator<uint> last = first + tmp_labels.size();

  auto z_iter = thrust::make_zip_iterator(thrust::make_tuple(
    first, t_label_roots + (label_roots.size() - n_clusters)));

  thrust::for_each(z_iter, z_iter + n_clusters,
                   init_label_roots<value_idx>(tmp_labels.data()));

  /**
   * Step 2: Propagate labels by having children iterate through their parents
   *     1. Initialize labels to -1
   *     2. For each element in levels array, propagate until parent's
   *        label is !=-1
   */

  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaGetLastError());

  value_idx cut_level = (children.size() / 2) - (n_clusters - 1);

  inherit_labels<<<n_blocks, 1024, 0, stream>>>(
    children.data(), levels.data(), n_leaves, tmp_labels.data(), cut_level);

  // copy tmp labels to actual labels
  raft::copy_async(labels, tmp_labels.data(), n_leaves, stream);
}

};  // end namespace Agglomerative
};  // end namespace Label
};  // end namespace Linkage
};  // end namespace ML
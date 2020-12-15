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

#include <raft/mr/device/buffer.hpp>

namespace ML {
namespace Linkage {
namespace Label {
namespace Agglomerative {

template <typename value_idx, typename value_t>
struct UnionFind {
  value_idx *parent_arr;
  value_idx *size_arr;
  value_idx next_label;
  value_idx *parent;
  value_idx *size;

  UnionFind(value_idx N_) {
    parent = new value_idx[2 * N_ - 1];
    next_label = N_;
    size = new value_idx[2 * N_ - 1];

    for (int i = 0; i < 2 * N_ - 1; i++) {
      parent[i] = -1;
      size[i] = i < N_ ? 1 : 0;
    }
  }

  value_idx find(value_idx n) {
    value_idx p;
    p = n;

    while (parent[n] != -1) n = parent[n];

    while (parent[p] != n) {
      p = parent[p];
      parent[p] = n;
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
void label_hierarchy_host(const raft::handle_t &handle,
                          const value_idx *rows,
                          const value_idx *cols,
                          const value_t *data,
                          size_t nnz,
                          value_idx *children,
                          value_t *out_delta,
                          value_idx *out_size) {
  auto stream = handle.get_stream();

  value_idx n_edges = nnz;

  CUML_LOG_INFO("Copying to host");

  std::vector<value_idx> mst_src_h(n_edges);
  std::vector<value_idx> mst_dst_h(n_edges);
  std::vector<value_t> mst_weights_h(n_edges);

  std::vector<value_idx> children_h(n_edges*2);
  std::vector<value_t> out_delta_h(n_edges);
  std::vector<value_idx> out_size_h(n_edges);

  raft::update_host(mst_src_h.data(), rows, n_edges, stream);
  raft::update_host(mst_dst_h.data(), cols, n_edges, stream);
  raft::update_host(mst_weights_h.data(), data, n_edges, stream);

  raft::update_host(children_h.data(), children, n_edges*2, stream);
  raft::update_host(out_delta_h.data(), out_delta, n_edges, stream);
  raft::update_host(out_size_h.data(), out_size, n_edges, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUML_LOG_INFO("Labeling");

  value_idx a, aa, b, bb;
  value_t delta;

  value_idx N = nnz + 1;

  CUML_LOG_INFO("Creating union find");

  UnionFind<value_idx, value_t> U(N);

  CUML_LOG_INFO("Done.");

  for (int i = 0; i < n_edges; i++) {

    a = mst_src_h.data()[i];
    b = mst_dst_h.data()[i];

    delta = mst_weights_h.data()[i];

    printf("a=%d, b=%d, delta=%f\n", a, b, delta);

    aa = U.find(a);
    bb = U.find(b);

    printf("a=%d, b=%d, delta=%f, aa=%d, bb=%d", a, b, delta, aa, bb);

    int children_idx = i * 2;

    children_h[children_idx] = aa;
    children_h[children_idx+1] = bb;
    out_delta_h[i] = delta;
    out_size_h[i] = U.size[aa] + U.size[b];

    U.perform_union(aa, bb);
  }

  CUML_LOG_INFO("Copying back to device");

  raft::update_device(children, children_h.data(), n_edges, stream);
  raft::update_device(out_delta, out_delta_h.data(), n_edges, stream);
  raft::update_device(out_size, out_size_h.data(), n_edges, stream);
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
void label_hierarchy_device(const raft::handle_t &handle,
                            const value_idx *rows,
                            const value_idx *cols,
                            const value_t *data,
                            value_idx nnz,
                            value_idx *children,
                            value_t *out_delta,
                            value_idx *out_size,
                            value_idx k_folds) {
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

template <typename value_idx, typename value_t>
void bfs_from_hierarchy() {}

template <typename value_idx, typename value_t>
void condense(value_idx *tree_src, value_idx *tree_dst, value_t *tree_delta,
              value_idx *tree_size, value_idx m) {
  value_idx root = 2 * m;

  value_idx n_points = root / 2 + 1;
  value_idx next_label = n_points + 1;
}

};  // end namespace Agglomerative
};  // end namespace Label
};  // end namespace Linkage
};  // end namespace ML
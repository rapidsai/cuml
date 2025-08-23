/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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

#include "kernels/condense.cuh"

#include <cuml/cluster/hdbscan.hpp>

#include <raft/sparse/convert/csr.cuh>
#include <raft/sparse/op/sort.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/cub.cuh>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>

#include <fstream>

namespace ML {
namespace HDBSCAN {
namespace detail {
namespace Condense {

/**
 * Condenses a binary single-linkage tree dendrogram in the Scipy hierarchy
 * format by collapsing subtrees that fall below a minimum cluster size.
 *
 * For increased parallelism, the output array sizes are held fixed but
 * the result will be sparse (e.g. zeros in place of parents who have been
 * removed / collapsed). This function accepts an empty instance of
 * `CondensedHierarchy` and invokes the `condense()` function on it to
 * convert the sparse output arrays into their dense form.
 *
 * @tparam value_idx
 * @tparam value_t
 * @tparam tpb
 * @param handle
 * @param[in] children parents/children from single-linkage dendrogram
 * @param[in] delta distances from single-linkage dendrogram
 * @param[in] sizes sizes from single-linkage dendrogram
 * @param[in] min_cluster_size any subtrees less than this size will be
 *                             collapsed.
 * @param[in] n_leaves number of actual data samples in the dendrogram
 * @param[out] condensed_tree output dendrogram. will likely no longer be
 *                            a binary tree.
 */
template <typename value_idx, typename value_t, int tpb = 256>
void build_condensed_hierarchy(const raft::handle_t& handle,
                               const value_idx* children,
                               const value_t* delta,
                               const value_idx* sizes,
                               int min_cluster_size,
                               int n_leaves,
                               Common::CondensedHierarchy<value_idx, value_t>& condensed_tree)
{
  cudaStream_t stream = handle.get_stream();
  auto exec_policy    = handle.get_thrust_policy();

  // Root is the last edge in the dendrogram
  value_idx root = 2 * (n_leaves - 1);

  auto d_ptr           = thrust::device_pointer_cast(children);
  value_idx n_vertices = *(thrust::max_element(exec_policy, d_ptr, d_ptr + root)) + 1;

  // Prevent potential infinite loop from labeling disconnected
  // connectivities graph.
  RAFT_EXPECTS(n_vertices == root,
               "Multiple components found in MST or MST is invalid. "
               "Cannot find single-linkage solution. Found %d vertices "
               "total.",
               static_cast<int>(n_vertices));

  rmm::device_uvector<bool> frontier(root + 1, stream);
  rmm::device_uvector<bool> next_frontier(root + 1, stream);

  thrust::fill(exec_policy, frontier.begin(), frontier.end(), false);
  thrust::fill(exec_policy, next_frontier.begin(), next_frontier.end(), false);

  // Array to propagate the lambda of subtrees actively being collapsed
  // through multiple bfs iterations.
  rmm::device_uvector<value_t> ignore(root + 1, stream);

  // Propagate labels from root
  rmm::device_uvector<value_idx> relabel(root + 1, handle.get_stream());
  thrust::fill(exec_policy, relabel.begin(), relabel.end(), -1);

  raft::update_device(relabel.data() + root, &root, 1, handle.get_stream());

  // Flip frontier for root
  constexpr bool start = true;
  raft::update_device(frontier.data() + root, &start, 1, handle.get_stream());

  rmm::device_uvector<value_idx> out_parent((root + 1) * 2, stream);
  rmm::device_uvector<value_idx> out_child((root + 1) * 2, stream);
  rmm::device_uvector<value_t> out_lambda((root + 1) * 2, stream);
  rmm::device_uvector<value_idx> out_size((root + 1) * 2, stream);

  thrust::fill(exec_policy, out_parent.begin(), out_parent.end(), -1);
  thrust::fill(exec_policy, out_child.begin(), out_child.end(), -1);
  thrust::fill(exec_policy, out_lambda.begin(), out_lambda.end(), -1);
  thrust::fill(exec_policy, out_size.begin(), out_size.end(), -1);
  thrust::fill(exec_policy, ignore.begin(), ignore.end(), -1);

  // While frontier is not empty, perform single bfs through tree
  size_t grid = raft::ceildiv(root + 1, static_cast<value_idx>(tpb));
  value_idx n_elements_to_traverse =
    thrust::reduce(exec_policy, frontier.data(), frontier.data() + root + 1, 0);

  int count = 0;
  while (n_elements_to_traverse > 0) {
    count += 1;
    // TODO: Investigate whether it would be worth performing a gather/argmatch in order
    // to schedule only the number of threads needed. (it might not be worth it)
    condense_hierarchy_kernel<<<grid, tpb, 0, handle.get_stream()>>>(frontier.data(),
                                                                     next_frontier.data(),
                                                                     ignore.data(),
                                                                     relabel.data(),
                                                                     children,
                                                                     delta,
                                                                     sizes,
                                                                     n_leaves,
                                                                     min_cluster_size,
                                                                     out_parent.data(),
                                                                     out_child.data(),
                                                                     out_lambda.data(),
                                                                     out_size.data());

    thrust::copy(exec_policy, next_frontier.begin(), next_frontier.end(), frontier.begin());
    thrust::fill(exec_policy, next_frontier.begin(), next_frontier.end(), false);

    n_elements_to_traverse = thrust::reduce(
      exec_policy, frontier.data(), frontier.data() + root + 1, static_cast<value_idx>(0));

    handle.sync_stream(stream);
    // std::cout << "\t\trun condense_hierarchy_kernel num elems to traverse: "  <<
    // n_elements_to_traverse  << std::endl;
  }
  std::cout << "sync twice run the while loop " << count << " times\n";

  //   // ---- save to disk for Python ----
  // {
  //   size_t size_used = (root + 1) * 2;
  //   std::vector<value_idx> out_parent_h((root + 1) * 2);
  //   std::vector<value_idx> out_child_h((root + 1) * 2);
  //   std::vector<value_t> out_lambda_h((root + 1) * 2);
  //   std::vector<value_idx> out_size_h((root + 1) * 2);

  //   raft::copy(out_parent_h.data(), out_parent.data(), size_used, stream);
  //   raft::copy(out_child_h.data(), out_child.data(), size_used, stream);
  //   raft::copy(out_lambda_h.data(), out_lambda.data(), size_used, stream);
  //   raft::copy(out_size_h.data(), out_size.data(), size_used, stream);

  //   std::cout << "saving information after condense to disk\n";
  //   std::ofstream f1("bc_out_parent_h_1000.bin", std::ios::binary);
  //   f1.write((char*)out_parent_h.data(), out_parent_h.size() * sizeof(value_idx));
  //   f1.close();

  //   std::ofstream f2("bc_out_child_h_1000.bin", std::ios::binary);
  //   f2.write((char*)out_child_h.data(), out_child_h.size() * sizeof(value_idx));
  //   f2.close();

  //   std::ofstream f3("bc_out_lambda_h_1000.bin", std::ios::binary);
  //   f3.write((char*)out_lambda_h.data(), out_lambda_h.size() * sizeof(value_t));
  //   f3.close();

  //   std::ofstream f4("bc_out_size_h_1000.bin", std::ios::binary);
  //   f4.write((char*)out_size_h.data(), out_size_h.size() * sizeof(value_idx));
  //   f4.close();
  // }
  // //  ----------------------------
  condensed_tree.condense(out_parent.data(), out_child.data(), out_lambda.data(), out_size.data());
}

// // Node struct for pointer chasing tree
// template <typename value_idx, typename value_t>
// struct Node {
//     value_idx idx;
//     value_idx size;
//     value_t lambda_val;
//     Node* parent;
//     Node* first_child;
//     Node* next_sibling;

//     Node(value_idx i)
//         : idx(i),
//           size(1),
//           lambda_val(std::numeric_limits<value_t>::infinity()),
//           parent(nullptr),
//           first_child(nullptr),
//           next_sibling(nullptr) {}
// };

// template <typename value_idx, typename value_t, int tpb = 256>
// void build_condensed_hierarchy(const raft::handle_t& handle,
//                                const value_idx* children,   // shape (n_leaves - 1, 2)
//                                const value_t* delta,        // shape (n_leaves - 1)
//                                const value_idx* sizes,      // shape (n_leaves - 1)
//                                int min_cluster_size,
//                                int n_leaves,
//                                Common::CondensedHierarchy<value_idx, value_t>& condensed_tree)
// {
//     int n_merges = n_leaves - 1;
//     int n_nodes  = n_leaves + n_merges;

//     // Allocate nodes
//     std::vector<Node<value_idx, value_t>*> nodes;
//     nodes.reserve(n_nodes);
//     for (int i = 0; i < n_nodes; i++) {
//         nodes.push_back(new Node<value_idx, value_t>(i));
//     }

//     // Build tree with pointer chasing
//     for (int t = 0; t < n_merges; t++) {
//         value_idx parent = n_leaves + t;
//         value_idx left   = children[2 * t];
//         value_idx right  = children[2 * t + 1];
//         value_t lambda_val = delta[t];

//         nodes[parent]->size = sizes[t];
//         nodes[parent]->lambda_val = lambda_val;

//         nodes[left]->parent  = nodes[parent];
//         nodes[right]->parent = nodes[parent];

//         nodes[parent]->first_child = nodes[left];
//         nodes[left]->next_sibling  = nodes[right];
//     }

//     Node<value_idx,value_t>* root = nodes.back();

//     // Shared output vectors (protected by OpenMP locks or per-thread buffers)
//     std::vector<value_idx> out_parent;
//     std::vector<value_idx> out_child;
//     std::vector<value_t>   out_lambda;
//     std::vector<value_idx> out_size;

//     // Use a lock-free approach with thread-local buffers, then merge
//     int n_threads = omp_get_max_threads();
//     std::vector<std::vector<value_idx>> local_parent(n_threads);
//     std::vector<std::vector<value_idx>> local_child(n_threads);
//     std::vector<std::vector<value_t>>   local_lambda(n_threads);
//     std::vector<std::vector<value_idx>> local_size(n_threads);

//     // Parallel DFS using explicit stack
//     #pragma omp parallel
//     {
//         int tid = omp_get_thread_num();
//         std::vector<Node<value_idx,value_t>*> stack;

//         #pragma omp single nowait
//         {
//             stack.push_back(root);
//         }

//         while (true) {
//             Node<value_idx,value_t>* node = nullptr;

//             // Critical section for shared stack
//             #pragma omp critical
//             {
//                 if (!stack.empty()) {
//                     node = stack.back();
//                     stack.pop_back();
//                 }
//             }

//             if (!node) {
//                 // No more work for this thread
//                 break;
//             }

//             // Iterate children
//             Node<value_idx,value_t>* child = node->first_child;
//             while (child) {
//                 if (child->size >= min_cluster_size) {
//                     local_parent[tid].push_back(node->idx);
//                     local_child[tid].push_back(child->idx);
//                     local_lambda[tid].push_back(node->lambda_val);
//                     local_size[tid].push_back(child->size);

//                     // Add child to stack for further traversal
//                     #pragma omp critical
//                     stack.push_back(child);
//                 }
//                 child = child->next_sibling;
//             }
//         }
//     }

//     // Merge thread-local buffers
//     for (int t = 0; t < n_threads; t++) {
//         out_parent.insert(out_parent.end(),
//                           local_parent[t].begin(), local_parent[t].end());
//         out_child.insert(out_child.end(),
//                          local_child[t].begin(), local_child[t].end());
//         out_lambda.insert(out_lambda.end(),
//                           local_lambda[t].begin(), local_lambda[t].end());
//         out_size.insert(out_size.end(),
//                         local_size[t].begin(), local_size[t].end());
//     }

//     // Push into condensed_tree
//     condensed_tree.condense(out_parent.data(),
//                             out_child.data(),
//                             out_lambda.data(),
//                             out_size.data(),
//                             static_cast<value_idx>(out_parent.size()));

//     // Cleanup
//     for (auto n : nodes) delete n;
// }

};  // end namespace Condense
};  // end namespace detail
};  // end namespace HDBSCAN
};  // end namespace ML

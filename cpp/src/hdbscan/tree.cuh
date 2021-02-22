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

#include <cuml/cuml_api.h>
#include <raft/cudart_utils.h>
#include <common/cumlHandle.hpp>

#include <raft/mr/device/buffer.hpp>
#include <raft/sparse/mst/mst.cuh>

#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <distance/distance.cuh>

#include <cuml/neighbors/knn.hpp>

namespace ML {
namespace HDBSCAN {
namespace Condense {

template <typename value_idx, typename value_t>
void bfs_from_hierarchy(const std::vector<value_idx> &src,
                        const std::vector<value_idx> &dst,
                        const std::vector<value_t> &delta,
                        const std::vector<value_idx> &size, value_idx bfs_root,
                        std::vector<value_idx> &result) {
  std::vector<value_idx> to_process;
  to_process.push_back(bfs_root);

  value_idx max_node;
  value_idx num_points;
  value_idx dim;

  dim = children.size();
  max_node = 2 * dim;
  num_points = max_node - dim + 1;

  while (to_process.size() > 0) {
    result.insert(result.end(), to_process.begin(), to_process.end());

    to_process.erase(
      std::remove_if(std::begin(to_process), std::end(to_process),
                     [&](const auto &elem) { return elem >= num_points; }));

    std::for_each(to_process.begin(), to_process.end(),
                  [&](const auto &elem) { return elem - num_points; });

    if (to_process.size() > 0) {
      // TODO: Get all items from to_process
      // and pull back children from
    }

    to_process = hierarchy [to_process, :2].flatten().astype(np.intp).tolist()
  }
  return result
}

template <typename value_idx, typename value_t>
void condense_host(value_idx *tree_src, value_idx *tree_dst,
                   value_t *tree_delta, value_idx *tree_size, value_idx m) {
  value_idx root = 2 * m;

  value_idx num_points = root / 2 + 1;
  value_idx next_label = num_points + 1;

  std::vector<value_idx> node_list;
  std::vector<value_idx> result_list_parent;
  std::vector<value_idx> result_list_child;
  std::vector<value_idx> result_list_lambda_val;
  std::vector<value_idx> result_list_child_size;

  std::vector<value_t> children;

  value_idx relabel[root + 1];
  relabel[root] = num_points;

  bool ignore[m];

  value_idx node;
  value_idx sub_node;
  value_idx left;
  value_idx right;
  value_t lambda_value;
  value_idx left_count;
  value_idx right_count;

  bfs_from_hierarchy(hierarchy, root, node_list);

  std::for_each(node_list.begin(), node_list.end(), [&](const auto &elem) {}) {
    if (!ignore[node] && node >= num_points) {
      children = hierarchy[node - num_points]
      left = <np.intp_t> children[0]
      right = <np.intp_t> children[1]
      if children[2] > 0.0:
      lambda_value = 1.0 / children[2]
      else:
      lambda_value = INFTY

      if left >= num_points:
      left_count = <np.intp_t> hierarchy[left - num_points][3]
      else:
      left_count = 1

      if right >= num_points:
      right_count = <np.intp_t> hierarchy[right - num_points][3]
      else:
      right_count = 1

      if left_count >= min_cluster_size and right_count >= min_cluster_size:
      relabel[left] = next_label
      next_label += 1
      result_list.append((relabel[node], relabel[left], lambda_value,
        left_count))

      relabel[right] = next_label
      next_label += 1
      result_list.append((relabel[node], relabel[right], lambda_value,
        right_count))

      elif left_count < min_cluster_size and right_count < min_cluster_size:
      for sub_node in bfs_from_hierarchy(hierarchy, left):
      if sub_node < num_points:
      result_list.append((relabel[node], sub_node,
        lambda_value, 1))
      ignore[sub_node] = True

      for sub_node in bfs_from_hierarchy(hierarchy, right):
      if sub_node < num_points:
      result_list.append((relabel[node], sub_node,
        lambda_value, 1))
      ignore[sub_node] = True

      elif left_count < min_cluster_size:
      relabel[right] = relabel[node]
      for sub_node in bfs_from_hierarchy(hierarchy, left):
      if sub_node < num_points:
      result_list.append((relabel[node], sub_node,
        lambda_value, 1))
      ignore[sub_node] = True

      else:
      relabel[left] = relabel[node]
      for sub_node in bfs_from_hierarchy(hierarchy, right):
      if sub_node < num_points:
      result_list.append((relabel[node], sub_node,
        lambda_value, 1))
      ignore[sub_node] = True
    }
  });

  return np.array(result_list, dtype = [
    ('parent', np.intp), ('child', np.intp), ('lambda_val', float),
    ('child_size', np.intp)
  ])
}

};  // end namespace Condense
};  // end namespace HDBSCAN
};  // end namespace ML
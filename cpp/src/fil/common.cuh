/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

/** @file common.cuh Common GPU functionality */

#include <stdio.h>
#include <stdlib.h>
#include <cub/cub.cuh>
#include <stdexcept>
#include <string>

#include "../../src_prims/cuda_utils.h"

#include <cuml/fil/fil.h>

namespace ML {
namespace fil {

__host__ __device__ __forceinline__ int tree_num_nodes(int depth) {
  return (1 << (depth + 1)) - 1;
}

__host__ __device__ __forceinline__ int forest_num_nodes(int num_trees,
                                                         int depth) {
  return num_trees * tree_num_nodes(depth);
}

// FIL_TPB is the number of threads per block to use with FIL kernels
const int FIL_TPB = 256;

/** base_node contains common implementation details for dense and sparse nodes */
struct base_node {
  static const int FID_MASK = (1 << 30) - 1;
  static const int DEF_LEFT_MASK = 1 << 30;
  static const int IS_LEAF_MASK = 1 << 31;
  float val;
  int bits;
  __host__ __device__ float output() const { return val; }
  __host__ __device__ float thresh() const { return val; }
  __host__ __device__ int fid() const { return bits & FID_MASK; }
  __host__ __device__ bool def_left() const { return bits & DEF_LEFT_MASK; }
  __host__ __device__ bool is_leaf() const { return bits & IS_LEAF_MASK; }
  __host__ __device__ base_node() : val(0.0f), bits(0) {}
  base_node(dense_node_t node) : val(node.val), bits(node.bits) {}
  base_node(float output, float thresh, int fid, bool def_left, bool is_leaf)
    : val(is_leaf ? output : thresh),
      bits((fid & FID_MASK) | (def_left ? DEF_LEFT_MASK : 0) |
           (is_leaf ? IS_LEAF_MASK : 0)) {}
};

/** dense_node is a single node of a dense forest */
struct alignas(8) dense_node : base_node {
  __host__ __device__ dense_node() : base_node() {}
  dense_node(dense_node_t node) : base_node(node) {}
  dense_node(float output, float thresh, int fid, bool def_left, bool is_leaf)
    : base_node(output, thresh, fid, def_left, is_leaf) {}
  /** index of the left child, where curr is the index of the current node */
  __host__ __device__ int left(int curr) const { return 2 * curr + 1; }
};

/** dense_tree represents a dense tree */
struct dense_tree {
  __host__ __device__ dense_tree(dense_node* nodes, int node_pitch)
    : nodes_(nodes), node_pitch_(node_pitch) {}
  __host__ __device__ const dense_node& operator[](int i) const {
    return nodes_[i * node_pitch_];
  }
  dense_node* nodes_ = nullptr;
  int node_pitch_ = 0;
};

/** dense_storage stores the forest as a collection of dense nodes */
struct dense_storage {
  __host__ __device__ dense_storage(dense_node* nodes, int num_trees,
                                    int tree_stride, int node_pitch)
    : nodes_(nodes),
      num_trees_(num_trees),
      tree_stride_(tree_stride),
      node_pitch_(node_pitch) {}
  __host__ __device__ int num_trees() const { return num_trees_; }
  __host__ __device__ dense_tree operator[](int i) const {
    return dense_tree(nodes_ + i * tree_stride_, node_pitch_);
  }
  dense_node* nodes_ = nullptr;
  int num_trees_ = 0;
  int tree_stride_ = 0;
  int node_pitch_ = 0;
};

/** sparse_node is a single node in a sparse forest */
struct alignas(16) sparse_node : base_node {
  int left_idx;
  // pad the size to 16 bytes to match sparse_node_t (in fil.h)
  int dummy;
  __host__ __device__ sparse_node() : left_idx(0), base_node() {}
  sparse_node(sparse_node_t node)
    : base_node(dense_node_t{node.val, node.bits}), left_idx(node.left_idx) {}
  sparse_node(float output, float thresh, int fid, bool def_left, bool is_leaf,
              int left_index)
    : base_node(output, thresh, fid, def_left, is_leaf), left_idx(left_index) {}
  __host__ __device__ int left_index() const { return left_idx; }
  /** index of the left child, where curr is the index of the current node */
  __host__ __device__ int left(int curr) const { return left_idx; }
};

/** sparse_tree is a sparse tree */
struct sparse_tree {
  __host__ __device__ sparse_tree(sparse_node* nodes) : nodes_(nodes) {}
  __host__ __device__ const sparse_node& operator[](int i) const {
    return nodes_[i];
  }
  sparse_node* nodes_ = nullptr;
};

/** sparse_storage stores the forest as a collection of sparse nodes */
struct sparse_storage {
  int* trees_ = nullptr;
  sparse_node* nodes_ = nullptr;
  int num_trees_ = 0;
  __host__ __device__ sparse_storage(int* trees, sparse_node* nodes,
                                     int num_trees)
    : trees_(trees), nodes_(nodes), num_trees_(num_trees) {}
  __host__ __device__ int num_trees() const { return num_trees_; }
  __host__ __device__ sparse_tree operator[](int i) const {
    return sparse_tree(&nodes_[trees_[i]]);
  }
};

// predict_params are parameters for prediction
struct predict_params {
  // Model parameters.
  int num_cols;
  algo_t algo;
  int max_items;  // only set and used by infer()
  int num_output_classes;
  // so far, only 1 or 2 is supported, and only used to output probabilities
  // from classifier models

  // Data parameters.
  float* preds;
  const float* data;
  size_t num_rows;

  // Other parameters.
  int max_shm;
};

// infer() calls the inference kernel with the parameters on the stream
template <typename storage_type>
void infer(storage_type forest, predict_params params, cudaStream_t stream);

}  // namespace fil
}  // namespace ML

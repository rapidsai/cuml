/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cub/cub.cuh>
#include <stdexcept>
#include <string>

#include <cuml/fil/fil.h>
#include <raft/cuda_utils.cuh>

#include "internal.cuh"

namespace ML {
namespace fil {

__host__ __device__ __forceinline__ int tree_num_nodes(int depth) { return (1 << (depth + 1)) - 1; }

__host__ __device__ __forceinline__ int forest_num_nodes(int num_trees, int depth)
{
  return num_trees * tree_num_nodes(depth);
}

/** dense_tree represents a dense tree */

struct dense_tree : categorical_branches {
  __host__ __device__ dense_tree(categorical_branches cat_branches,
                                 dense_node* nodes,
                                 int node_pitch)
    : categorical_branches(cat_branches), nodes_(nodes), node_pitch_(node_pitch)
  {
  }
  __host__ __device__ const dense_node& operator[](int i) const { return nodes_[i * node_pitch_]; }
  dense_node* nodes_ = nullptr;
  int node_pitch_    = 0;
};

/** dense_storage stores the forest as a collection of dense nodes */

struct dense_storage : categorical_branches {
  __host__ __device__ dense_storage(dense_node* nodes,
                                    int num_trees,
                                    int tree_stride,
                                    int node_pitch,
                                    float* vector_leaf,
                                    categorical_branches cat_branches)
    : categorical_branches(cat_branches),
      nodes_(nodes),
      num_trees_(num_trees),
      tree_stride_(tree_stride),
      node_pitch_(node_pitch),
      vector_leaf_(vector_leaf)
  {
  }
  __host__ __device__ int num_trees() const { return num_trees_; }
  __host__ __device__ dense_tree operator[](int i) const
  {
    return dense_tree(*this, nodes_ + i * tree_stride_, node_pitch_);
  }
  dense_node* nodes_  = nullptr;
  float* vector_leaf_ = nullptr;
  int num_trees_      = 0;
  int tree_stride_    = 0;
  int node_pitch_     = 0;
};

/** sparse_tree is a sparse tree */
template <typename node_t>
struct sparse_tree : categorical_branches {
  __host__ __device__ sparse_tree(categorical_branches cat_branches, node_t* nodes)
    : categorical_branches(cat_branches), nodes_(nodes)
  {
  }
  __host__ __device__ const node_t& operator[](int i) const { return nodes_[i]; }
  node_t* nodes_ = nullptr;
};

/** sparse_storage stores the forest as a collection of sparse nodes */
template <typename node_t>
struct sparse_storage : categorical_branches {
  int* trees_         = nullptr;
  node_t* nodes_      = nullptr;
  float* vector_leaf_ = nullptr;
  int num_trees_      = 0;
  __host__ __device__ sparse_storage(
    int* trees, node_t* nodes, int num_trees, float* vector_leaf, categorical_branches cat_branches)
    : categorical_branches(cat_branches),
      trees_(trees),
      nodes_(nodes),
      num_trees_(num_trees),
      vector_leaf_(vector_leaf)
  {
  }
  __host__ __device__ int num_trees() const { return num_trees_; }
  __host__ __device__ sparse_tree<node_t> operator[](int i) const
  {
    return sparse_tree<node_t>(*this, &nodes_[trees_[i]]);
  }
};

typedef sparse_storage<sparse_node16> sparse_storage16;
typedef sparse_storage<sparse_node8> sparse_storage8;

/// all model parameters mostly required to compute shared memory footprint,
/// also the footprint itself
struct shmem_size_params {
  /// for class probabilities, this is the number of classes considered;
  /// num_classes is ignored otherwise
  int num_classes = 1;
  // leaf_algo determines what the leaves store (predict) and how FIL
  // aggregates them into class margins/predicted class/regression answer
  leaf_algo_t leaf_algo = leaf_algo_t::FLOAT_UNARY_BINARY;
  /// how many columns an input row has
  int num_cols = 0;
  /// whether to predict class probabilities or classes (or regress)
  bool predict_proba = false;
  /// are the input columns are prefetched into shared
  /// memory before inferring the row in question
  bool cols_in_shmem = true;
  /// log2_threads_per_tree determines how many threads work on a single tree
  /// at once inside a block (sharing trees means splitting input rows)
  int log2_threads_per_tree = 0;
  /// n_items is how many input samples (items) any thread processes. If 0 is given,
  /// choose the reasonable most (<=4) that fit into shared memory. See init_n_items()
  int n_items = 0;
  /// shm_sz is the associated shared memory footprint
  int shm_sz = INT_MAX;

  __host__ __device__ int sdata_stride()
  {
    return num_cols | 1;  // pad to odd
  }
  __host__ __device__ int cols_shmem_size()
  {
    return cols_in_shmem ? sizeof(float) * sdata_stride() * n_items << log2_threads_per_tree : 0;
  }
  void compute_smem_footprint();
  template <int NITEMS>
  size_t get_smem_footprint();
  template <int NITEMS, leaf_algo_t leaf_algo>
  size_t get_smem_footprint();
};

// predict_params are parameters for prediction
struct predict_params : shmem_size_params {
  predict_params(shmem_size_params ssp) : shmem_size_params(ssp) {}
  // Model parameters.
  algo_t algo;
  // number of outputs for the forest per each data row
  int num_outputs;

  // Data parameters.
  float* preds;
  const float* data;
  // number of data rows (instances) to predict on
  int64_t num_rows;

  // to signal infer kernel to apply softmax and also average prior to that
  // for GROVE_PER_CLASS for predict_proba
  output_t transform;
  // number of blocks to launch
  int num_blocks;
  // can a node be categorical in this forest?
  bool can_be_categorical;
};

// infer() calls the inference kernel with the parameters on the stream
template <typename storage_type>
void infer(storage_type forest, predict_params params, cudaStream_t stream);

}  // namespace fil
}  // namespace ML

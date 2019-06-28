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

#include "fil.h"

namespace ML {
namespace fil {

__host__ __device__ __forceinline__ int tree_num_nodes(int depth) {
  return (1 << (depth + 1)) - 1;
}

__host__ __device__ __forceinline__ int forest_num_nodes(int ntrees,
                                                         int depth) {
  return ntrees * tree_num_nodes(depth);
}

// TPB is the number of threads per block to use with FIL kernels
const int TPB = 256;

/** node is a single tree node. */
struct __align__(8) dense_node {
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
  __host__ __device__ dense_node() : val(0.0f), bits(0) {}
  dense_node(dense_node_t n) : val(n.val), bits(n.bits) {}
  dense_node(float output, float thresh, int fid, bool def_left, bool is_leaf)
    : val(is_leaf ? output : thresh),
      bits((fid & FID_MASK) | (def_left ? DEF_LEFT_MASK : 0) |
           (is_leaf ? IS_LEAF_MASK : 0)) {}
};

// predict_params are parameters for prediction
struct predict_params {
  // Forest parameters.
  const dense_node* nodes;
  int ntrees;
  int depth;
  int cols;

  // Data parameters.
  float* preds;
  const float* data;
  size_t rows;

  // Other parameters.
  int max_shm;
};

}  // namespace fil
}  // namespace ML

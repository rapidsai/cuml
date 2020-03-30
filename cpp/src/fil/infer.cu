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

#include <algorithm>
#include "common.cuh"

namespace ML {
namespace fil {

using namespace MLCommon;

// vec wraps float[N] for cub::BlockReduce
template <int N>
struct vec {
  float data[N];
  __host__ __device__ float& operator[](int i) { return data[i]; }
  __host__ __device__ float operator[](int i) const { return data[i]; }
  friend __host__ __device__ vec<N> operator+(const vec<N>& a,
                                              const vec<N>& b) {
    vec<N> r;
#pragma unroll
    for (int i = 0; i < N; ++i) r[i] = a[i] + b[i];
    return r;
  }
};

template <int NITEMS, typename tree_type>
__device__ __forceinline__ void infer_one_tree(tree_type tree, float* sdata,
                                               int cols, vec<NITEMS>& out) {
  int curr[NITEMS];
  int mask = (1 << NITEMS) - 1;  // all active
  for (int j = 0; j < NITEMS; ++j) curr[j] = 0;
  do {
#pragma unroll
    for (int j = 0; j < NITEMS; ++j) {
      if ((mask >> j) & 1 == 0) continue;
      auto n = tree[curr[j]];
      if (n.is_leaf()) {
        mask &= ~(1 << j);
        continue;
      }
      float val = sdata[j * cols + n.fid()];
      bool cond = isnan(val) ? !n.def_left() : val >= n.thresh();
      curr[j] = n.left(curr[j]) + cond;
    }
  } while (mask != 0);
#pragma unroll
  for (int j = 0; j < NITEMS; ++j) out[j] += tree[curr[j]].output();
}

template <typename tree_type>
__device__ __forceinline__ void infer_one_tree(tree_type tree, float* sdata,
                                               int cols, vec<1>& out) {
  int curr = 0;
  for (;;) {
    auto n = tree[curr];
    if (n.is_leaf()) break;
    float val = sdata[n.fid()];
    bool cond = isnan(val) ? !n.def_left() : val >= n.thresh();
    curr = n.left(curr) + cond;
  }
  out[0] = tree[curr].output();
}

template <int NITEMS, typename storage_type>
__global__ void infer_k(storage_type forest, predict_params params) {
  // cache the row for all threads to reuse
  extern __shared__ char smem[];
  float* sdata = (float*)smem;
  size_t rid = blockIdx.x * NITEMS;
  for (int j = 0; j < NITEMS; ++j) {
    for (int i = threadIdx.x; i < params.num_cols; i += blockDim.x) {
      size_t row = rid + j;
      sdata[j * params.num_cols + i] =
        row < params.num_rows ? params.data[row * params.num_cols + i] : 0.0f;
    }
  }
  __syncthreads();

  // one block works on NITEMS rows and the whole forest
  vec<NITEMS> out;
  for (int i = 0; i < NITEMS; ++i) out[i] = 0.0f;
  for (int j = threadIdx.x; j < forest.num_trees(); j += blockDim.x) {
    infer_one_tree<NITEMS>(forest[j], sdata, params.num_cols, out);
  }
  using BlockReduce = cub::BlockReduce<vec<NITEMS>, FIL_TPB>;
  __shared__ typename BlockReduce::TempStorage tmp_storage;
  out = BlockReduce(tmp_storage).Sum(out);
  if (threadIdx.x == 0) {
    for (int i = 0; i < NITEMS; ++i) {
      int row = blockIdx.x * NITEMS + i;
      if (row < params.num_rows)
        params.preds[row * params.num_output_classes] = out[i];
    }
  }
}

template <typename storage_type>
void infer(storage_type forest, predict_params params, cudaStream_t stream) {
  const int MAX_BATCH_ITEMS = 4;
  params.max_items =
    params.algo == algo_t::BATCH_TREE_REORG ? MAX_BATCH_ITEMS : 1;
  int num_items = params.max_shm / (sizeof(float) * params.num_cols);
  if (num_items == 0) {
    int max_cols = params.max_shm / sizeof(float);
    ASSERT(false, "p.num_cols == %d: too many features, only %d allowed",
           params.num_cols, max_cols);
  }
  num_items = std::min(num_items, params.max_items);
  int num_blocks = ceildiv(int(params.num_rows), num_items);
  int shm_sz = num_items * sizeof(float) * params.num_cols;
  switch (num_items) {
    case 1:
      infer_k<1><<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
      break;
    case 2:
      infer_k<2><<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
      break;
    case 3:
      infer_k<3><<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
      break;
    case 4:
      infer_k<4><<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
      break;
    default:
      ASSERT(false, "internal error: nitems > 4");
  }
  CUDA_CHECK(cudaPeekAtLastError());
}

template void infer<dense_storage>(dense_storage forest, predict_params params,
                                   cudaStream_t stream);
template void infer<sparse_storage>(sparse_storage forest,
                                    predict_params params, cudaStream_t stream);

}  // namespace fil
}  // namespace ML

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
template <int N, typename T>
struct vec {
  T data[N];
  __host__ __device__ T& operator[](int i) { return data[i]; }
  __host__ __device__ T operator[](int i) const { return data[i]; }
  friend __host__ __device__ void operator+=(vec<N, T>& a,
                                             const vec<N, T>& b) {
#pragma unroll
    for (int i = 0; i < N; ++i) a[i] += b[i];
  }
  friend __host__ __device__ vec<N, T> operator+(const vec<N, T>& a,
                                                 const vec<N, T>& b) {
    vec<N, T> r = a;
    r += b;
    return r;
  }                                               
};

template <int NITEMS, typename TOUTPUT, typename tree_type>
__device__ __forceinline__ vec<NITEMS, TOUTPUT> infer_one_tree(tree_type tree, float* sdata,
                                               int cols) {
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
  vec<NITEMS, TOUTPUT> out;
#pragma unroll
  for (int j = 0; j < NITEMS; ++j)
    out[j] = tree[curr[j]].base_node::output<TOUTPUT>();
  return out;
}

template <typename TOUTPUT, typename tree_type>
__device__ __forceinline__ vec<1, TOUTPUT> infer_one_tree(tree_type tree, float* sdata,
                                               int cols) {
  int curr = 0;
  for (;;) {
    auto n = tree[curr];
    if (n.is_leaf()) break;
    float val = sdata[n.fid()];
    bool cond = isnan(val) ? !n.def_left() : val >= n.thresh();
    curr = n.left(curr) + cond;
  }
  vec<1, TOUTPUT> out;
  out[0] = tree[curr].base_node::output<TOUTPUT>();
  return out;
}

template <int NITEMS, leaf_value_t leaf_payload_type, typename TNODE_PAYLOAD>
class AggregateTrees {
  public:
    __device__ __forceinline__ AggregateTrees(int num_output_classes, void* smem_workspace);
    __device__ __forceinline__ void accumulate(vec<NITEMS, TNODE_PAYLOAD> out);
    __device__ __forceinline__ void finalize(float* out, int num_rows);
};

template <int NITEMS> class AggregateTrees<NITEMS, FLOAT_SCALAR, float> {
  vec<NITEMS, float> acc;
  int num_output_classes;
  public:
    __device__ __forceinline__ AggregateTrees(int num_output_classes_, void*):
    num_output_classes(num_output_classes_) {
      // TODO: even if num_output_classes == 2, in regression, this needs to change
      #pragma unroll
      for (int i = 0; i < NITEMS; ++i) acc[i] = 0.0f;
    }
    __device__ __forceinline__ void accumulate(vec<NITEMS, float> out) {
      acc += out;
    }
    __device__ __forceinline__ void finalize(float* out, int num_rows) {
      using BlockReduce = cub::BlockReduce<vec<NITEMS, float>, FIL_TPB>;
      __shared__ typename BlockReduce::TempStorage tmp_storage;
      acc = BlockReduce(tmp_storage).Sum(acc);
      if (threadIdx.x == 0) {
        for (int i = 0; i < NITEMS; ++i) {
          int row = blockIdx.x * NITEMS + i;
          if (row < num_rows)
            out[row * num_output_classes] = acc[i];
        }
      }
    }
};

template <int NITEMS, leaf_value_t leaf_payload_type, typename TOUTPUT, typename storage_type>
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

  AggregateTrees<NITEMS, leaf_payload_type, TOUTPUT> acc(params.num_output_classes, nullptr);
  // one block works on NITEMS rows and the whole forest
  for (int j = threadIdx.x; j < forest.num_trees(); j += blockDim.x) {
    acc.accumulate(infer_one_tree<NITEMS, TOUTPUT>(forest[j], sdata, params.num_cols));
  }
  acc.finalize(params.preds, params.num_rows);
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
      switch (params.leaf_payload_type) {
        case FLOAT_SCALAR:
          ASSERT(params.num_output_classes <= 2, "wrong leaf payload for multi-class (>2) inference");
          infer_k<1, FLOAT_SCALAR, float><<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
          break;
        default:
          ASSERT(false, "only FLOAT_SCALAR supported as leaf_payload_type so far");
      }
      break;
    case 2:
      switch (params.leaf_payload_type) {
        case FLOAT_SCALAR:
          ASSERT(params.num_output_classes <= 2, "wrong leaf payload for multi-class (>2) inference");
          infer_k<2, FLOAT_SCALAR, float><<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
          break;
        default:
          ASSERT(false, "only FLOAT_SCALAR supported as leaf_payload_type so far");
      }
      break;
    case 3:
      switch (params.leaf_payload_type) {
        case FLOAT_SCALAR:
          ASSERT(params.num_output_classes <= 2, "wrong leaf payload for multi-class (>2) inference");
          infer_k<3, FLOAT_SCALAR, float><<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
          break;
        default:
          ASSERT(false, "only FLOAT_SCALAR supported as leaf_payload_type so far");
      }
      break;
    case 4:
      switch (params.leaf_payload_type) {
        case FLOAT_SCALAR:
          ASSERT(params.num_output_classes <= 2, "wrong leaf payload for multi-class (>2) inference");
          infer_k<4, FLOAT_SCALAR, float><<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
          break;
        default:
          ASSERT(false, "only FLOAT_SCALAR supported as leaf_payload_type so far");
      }
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

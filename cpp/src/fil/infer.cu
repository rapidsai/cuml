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
  inline __host__ __device__ vec() {
#pragma unroll
    for (int i = 0; i < N; ++i) data[i] = 0;
  }
  __host__ __device__ T& operator[](int i) { return data[i]; }
  __host__ __device__ T operator[](int i) const { return data[i]; }
  friend __host__ __device__ void operator+=(vec<N, T>& a, const vec<N, T>& b) {
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

template <int NITEMS, typename output_type, typename tree_type>
__device__ __forceinline__ vec<NITEMS, output_type> infer_one_tree(
  tree_type tree, float* sdata, int cols) {
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
  vec<NITEMS, output_type> out;
#pragma unroll
  for (int j = 0; j < NITEMS; ++j)
    out[j] = tree[curr[j]].base_node::output<output_type>();
  return out;
}

template <typename output_type, typename tree_type>
__device__ __forceinline__ vec<1, output_type> infer_one_tree(tree_type tree,
                                                              float* sdata,
                                                              int cols) {
  int curr = 0;
  for (;;) {
    auto n = tree[curr];
    if (n.is_leaf()) break;
    float val = sdata[n.fid()];
    bool cond = isnan(val) ? !n.def_left() : val >= n.thresh();
    curr = n.left(curr) + cond;
  }
  vec<1, output_type> out;
  out[0] = tree[curr].base_node::output<output_type>();
  return out;
}

template <int NITEMS,
          leaf_value_t leaf_payload_type>  // = FLOAT_SCALAR
struct tree_aggregator_t {
  vec<NITEMS, float> acc;

  __device__ __forceinline__ tree_aggregator_t(int, void*) {}
  __device__ __forceinline__ void accumulate(vec<NITEMS, float> out) {
    acc += out;
  }
  __device__ __forceinline__ void finalize(float* out, int num_rows,
                                           char output_stride) {
    __syncthreads();
    using BlockReduce = cub::BlockReduce<vec<NITEMS, float>, FIL_TPB>;
    __shared__ typename BlockReduce::TempStorage tmp_storage;
    acc = BlockReduce(tmp_storage).Sum(acc);
    if (threadIdx.x == 0) {
      for (int i = 0; i < NITEMS; ++i) {
        int row = blockIdx.x * NITEMS + i;
        if (row < num_rows) out[row * output_stride] = acc[i];
      }
    }
  }
};

template <int NITEMS>
struct tree_aggregator_t<NITEMS, INT_CLASS_LABEL> {
  typedef unsigned class_label_t;
  typedef unsigned vote_count_t;
  // could switch to unsigned short to save shared memory
  // provided atomicAdd(short*) simulated with appropriate shifts
  vote_count_t* votes;
  class_label_t num_classes;

  __device__ __forceinline__ tree_aggregator_t(int num_classes_,
                                               void* shared_workspace)
    : num_classes(num_classes_), votes((vote_count_t*)shared_workspace) {
    for (class_label_t c = threadIdx.x; c < num_classes; c += FIL_TPB * NITEMS)
#pragma unroll
      for (int item = 0; item < NITEMS; ++item) votes[c * NITEMS + item] = 0;
    //__syncthreads(); // happening outside already
  }
  __device__ __forceinline__ void accumulate(vec<NITEMS, unsigned> out) {
#pragma unroll
    for (int item = 0; item < NITEMS; ++item)
      atomicAdd(votes + out[item] * NITEMS + item, 1);
  }
  // class probabilities or regression. for regression, num_classes
  // is just the number of outputs for each data instance
  __device__ __forceinline__ void finalize_multiple_outputs(float* out,
                                                            int num_rows) {
    __syncthreads();
    int item = threadIdx.x;
    int row = blockIdx.x * NITEMS + item;
    if (item < NITEMS && row < num_rows) {
#pragma unroll
      for (int c = 0; c < num_classes; ++c)
        out[row * num_classes + c] = votes[c * NITEMS + item];
    }
  }
  // using this when predicting a single class label, as opposed to sparse class vector
  // or class probabilities or regression
  __device__ __forceinline__ void finalize_class_label(float* out,
                                                       int num_rows) {
    __syncthreads();
    int item = threadIdx.x;
    int row = blockIdx.x * NITEMS + item;
    if (item < NITEMS && row < num_rows) {
      vote_count_t max_votes = 0;
      class_label_t best_class = 0;
      for (int c = 0; c < num_classes; ++c)
        if (votes[c * NITEMS + item] > max_votes) {
          max_votes = votes[c * NITEMS + item];
          best_class = c;
        }
      out[row] = best_class;
    }
  }
  __device__ __forceinline__ void finalize(float* out, int num_rows,
                                           int num_outputs) {
    if (num_outputs > 1)
      // only supporting num_outputs == num_classes
      finalize_multiple_outputs(out, num_rows);
    else
      finalize_class_label(out, num_rows);
  }
};

template <int NITEMS, leaf_value_t leaf_payload_type, class storage_type>
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

  tree_aggregator_t<NITEMS, leaf_payload_type> acc(
    params.num_classes, sdata + params.num_cols * NITEMS);

  __syncthreads();  // for both row cache init and acc init

  AggregateTrees<NITEMS, leaf_payload_type, TOUTPUT> acc(params.num_output_classes, nullptr);
  // one block works on NITEMS rows and the whole forest
  for (int j = threadIdx.x; j < forest.num_trees(); j += blockDim.x) {
    acc.accumulate(infer_one_tree<NITEMS, leaf_output_t<leaf_payload_type>::T>(
      forest[j], sdata, params.num_cols));
  }
  acc.finalize(params.preds, params.num_rows, params.num_outputs);
}

template <leaf_value_t leaf_payload_type, typename storage_type>
void infer_k_launcher(storage_type forest, predict_params params,
                      cudaStream_t stream) {
  const int MAX_BATCH_ITEMS = 4;
  params.max_items =
    params.algo == algo_t::BATCH_TREE_REORG ? MAX_BATCH_ITEMS : 1;

  int shared_mem_per_item =
    sizeof(float) * params.num_cols +
    // class vote histogram, while inferring trees
    (leaf_payload_type == INT_CLASS_LABEL ? sizeof(int) * params.num_classes
                                          : 0);
  // CUB workspace should fit itself, and we don't need
  // the row by the time CUB is used
  int num_items = params.max_shm / shared_mem_per_item;
  if (num_items == 0) {
    int max_cols = params.max_shm / sizeof(float);
    ASSERT(false, "p.num_cols == %d: too many features, only %d allowed%s",
           params.num_cols, max_cols,
           leaf_payload_type == INT_CLASS_LABEL
             ? " (accounting for shared class vote histogram)"
             : "");
  }
  num_items = std::min(num_items, params.max_items);
  int num_blocks = ceildiv(int(params.num_rows), num_items);
  int shm_sz = num_items * shared_mem_per_item;
  switch (num_items) {
    case 1:
      infer_k<1, leaf_payload_type>
        <<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
      break;
    case 2:
      infer_k<2, leaf_payload_type>
        <<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
      break;
    case 3:
      infer_k<3, leaf_payload_type>
        <<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
      break;
    case 4:
      infer_k<4, leaf_payload_type>
        <<<num_blocks, FIL_TPB, shm_sz, stream>>>(forest, params);
      break;
    default:
      ASSERT(false, "internal error: nitems > 4");
  }
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename storage_type>
void infer(storage_type forest, predict_params params, cudaStream_t stream) {
  switch (params.leaf_payload_type) {
    case FLOAT_SCALAR:
      ASSERT(params.num_outputs <= 2,
             "wrong leaf payload for multi-class (>2) inference");
      infer_k_launcher<FLOAT_SCALAR, storage_type>(forest, params, stream);
      break;
    case INT_CLASS_LABEL:
      infer_k_launcher<INT_CLASS_LABEL, storage_type>(forest, params, stream);
      break;
    default:
      ASSERT(false, "unknown leaf_payload_type");
  }
}

template void infer<dense_storage>(dense_storage forest, predict_params params,
                                   cudaStream_t stream);
template void infer<sparse_storage>(sparse_storage forest,
                                    predict_params params, cudaStream_t stream);

}  // namespace fil
}  // namespace ML
#undef __forceinline__

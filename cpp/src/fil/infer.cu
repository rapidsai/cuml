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

template <int NITEMS>
__device__ __forceinline__ void infer_one_tree(const dense_node* root,
                                               float* sdata, int pitch,
                                               int cols, vec<NITEMS>& out) {
  int curr[NITEMS];
  int mask = (1 << NITEMS) - 1;  // all active
  for (int j = 0; j < NITEMS; ++j) curr[j] = 0;
  do {
#pragma unroll
    for (int j = 0; j < NITEMS; ++j) {
      if ((mask >> j) & 1 == 0) continue;
      dense_node n = root[curr[j] * pitch];
      if (n.is_leaf()) {
        mask &= ~(1 << j);
        continue;
      }
      float val = sdata[j * cols + n.fid()];
      bool cond = isnan(val) ? !n.def_left() : val >= n.thresh();
      curr[j] = (curr[j] << 1) + 1 + cond;
    }
  } while (mask != 0);
#pragma unroll
  for (int j = 0; j < NITEMS; ++j)
    out[j] += root[curr[j] * pitch].output();
}

__device__ __forceinline__ void infer_one_tree(const dense_node* root,
                                               float* sdata, int pitch,
                                               int cols, vec<1>& out) {
  int curr = 0;
  for (;;) {
    dense_node n = root[curr * pitch];
    if (n.is_leaf()) break;
    float val = sdata[n.fid()];
    bool cond = isnan(val) ? !n.def_left() : val >= n.thresh();
    curr = (curr << 1) + 1 + cond;
  }
  out[0] = root[curr * pitch].output();
}

template <int NITEMS>
__global__ void infer_k(predict_params ps) {
  // cache the row for all threads to reuse
  extern __shared__ char smem[];
  float* sdata = (float*)smem;
  size_t rid = blockIdx.x * NITEMS;
  for (int j = 0; j < NITEMS; ++j) {
    for (int i = threadIdx.x; i < ps.cols; i += blockDim.x) {
      size_t row = rid + j;
      sdata[j * ps.cols + i] =
        row < ps.rows ? ps.data[row * ps.cols + i] : 0.0f;
    }
  }
  __syncthreads();
  // one block works on a single row and the whole forest
  vec<NITEMS> out;
  for (int i = 0; i < NITEMS; ++i) out[i] = 0.0f;
  for (int j = threadIdx.x; j < ps.ntrees; j += blockDim.x) {
    infer_one_tree<NITEMS>(ps.nodes + j * ps.tree_stride, sdata, ps.pitch,
                           ps.cols, out);
  }
  typedef cub::BlockReduce<vec<NITEMS>, FIL_TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage tmp_storage;
  out = BlockReduce(tmp_storage).Sum(out);
  if (threadIdx.x == 0) {
    for (int i = 0; i < NITEMS; ++i) {
      int idx = blockIdx.x * NITEMS + i;
      if (idx < ps.rows) ps.preds[idx] = out[i];
    }
  }
}

void infer(predict_params ps, cudaStream_t stream) {
  const int MAX_BATCH_ITEMS = 4;
  ps.max_items = ps.algo == algo_t::BATCH_TREE_REORG ? MAX_BATCH_ITEMS : 1;
  ps.pitch = ps.algo == algo_t::NAIVE ? 1 : ps.ntrees;
  ps.tree_stride = ps.algo == algo_t::NAIVE ? tree_num_nodes(ps.depth) : 1;
  int num_items = ps.max_shm / (sizeof(float) * ps.cols);
  if (num_items == 0) {
    int max_cols = ps.max_shm / sizeof(float);
    ASSERT(false, "p.cols == %d: too many features, only %d allowed", ps.cols,
           max_cols);
  }
  num_items = std::min(num_items, ps.max_items);
  int nblks = ceildiv(int(ps.rows), num_items);
  int shm_sz = num_items * sizeof(float) * ps.cols;
  switch (num_items) {
    case 1:
      infer_k<1><<<nblks, FIL_TPB, shm_sz, stream>>>(ps);
      break;
    case 2:
      infer_k<2><<<nblks, FIL_TPB, shm_sz, stream>>>(ps);
      break;
    case 3:
      infer_k<3><<<nblks, FIL_TPB, shm_sz, stream>>>(ps);
      break;
    case 4:
      infer_k<4><<<nblks, FIL_TPB, shm_sz, stream>>>(ps);
      break;
    default:
      ASSERT(false, "internal error: nitems > 4");
  }
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace fil
}  // namespace ML

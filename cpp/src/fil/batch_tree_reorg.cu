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
__device__ __forceinline__ void infer_one_tree(const dense_node* root, int tree,
                                               float* sdata, int depth,
                                               int ntrees, int cols,
                                               vec<NITEMS>& out) {
  int curr[NITEMS];
  int mask = (1 << NITEMS) - 1;  // all active
  for (int j = 0; j < NITEMS; ++j) curr[j] = 0;
  do {
#pragma unroll
    for (int j = 0; j < NITEMS; ++j) {
      if ((mask >> j) & 1 == 0) continue;
      dense_node n = root[curr[j] * ntrees + tree];
      if (n.is_leaf()) {
        mask &= ~(1 << j);
        continue;
      }
      float val = sdata[j * cols + n.fid()];
      bool cond = isnan(val) ? !n.def_left() : val >= n.thresh();
      curr[j] = (curr[j] << 1) + 1 + cond;
    }
  } while (mask != 0);
  for (int j = 0; j < NITEMS; ++j)
    out[j] += root[curr[j] * ntrees + tree].output();
}

template <int NITEMS>
__global__ void batch_tree_reorg_kernel(predict_params ps) {
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
  int max_nodes = tree_num_nodes(ps.depth);
  for (int j = threadIdx.x; j < ps.ntrees; j += blockDim.x) {
    infer_one_tree<NITEMS>(ps.nodes, j, sdata, ps.depth, ps.ntrees, ps.cols,
                           out);
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

const static int MAX_NITEMS = 4;
void batch_tree_reorg(const predict_params& ps, cudaStream_t stream) {
  int nitems = ps.max_shm / (sizeof(float) * ps.cols);
  if (nitems == 0) {
    int max_cols = ps.max_shm / sizeof(float);
    ASSERT(false, "p.cols == %d: too many features, only %d allowed", ps.cols,
           max_cols);
  }
  nitems = std::min(nitems, MAX_NITEMS);
  int nblks = ceildiv(int(ps.rows), nitems);
  int shm_sz = nitems * sizeof(float) * ps.cols;
  switch (nitems) {
    case 1:
      batch_tree_reorg_kernel<1><<<nblks, FIL_TPB, shm_sz, stream>>>(ps);
      break;
    case 2:
      batch_tree_reorg_kernel<2><<<nblks, FIL_TPB, shm_sz, stream>>>(ps);
      break;
    case 3:
      batch_tree_reorg_kernel<3><<<nblks, FIL_TPB, shm_sz, stream>>>(ps);
      break;
    case 4:
      batch_tree_reorg_kernel<4><<<nblks, FIL_TPB, shm_sz, stream>>>(ps);
      break;
    default:
      ASSERT(false, "internal error: nitems > 4");
  }
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace fil
}  // namespace ML

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

#include "common.cuh"

namespace ML {
namespace fil {

__device__ __forceinline__ float infer_one_tree(const dense_node* root,
                                                float* sdata, int depth) {
  int curr = 0;
  for (;;) {
    dense_node n = root[curr];
    if (n.is_leaf()) break;
    float val = sdata[n.fid()];
    bool cond = isnan(val) ? !n.def_left() : val >= n.thresh();
    curr = (curr << 1) + 1 + cond;
  }
  return root[curr].output();
}

__global__ void naive_kernel(predict_params ps) {
  // cache the row for all threads to reuse
  extern __shared__ char smem[];
  float* sdata = (float*)smem;
  size_t rid = blockIdx.x;
  for (int i = threadIdx.x; i < ps.cols; i += blockDim.x)
    sdata[i] = ps.data[rid * ps.cols + i];
  __syncthreads();
  // one block works on a single row and the whole forest
  float out = 0.0f;
  int max_nodes = tree_num_nodes(ps.depth);
  for (int j = threadIdx.x; j < ps.ntrees; j += blockDim.x) {
    out += infer_one_tree(ps.nodes + j * max_nodes, sdata, ps.depth);
  }
  typedef cub::BlockReduce<float, TPB> BlockReduce;
  __shared__ BlockReduce::TempStorage tmp_storage;
  out = BlockReduce(tmp_storage).Sum(out);
  if (threadIdx.x == 0) ps.preds[blockIdx.x] = out;
}

void naive(const predict_params& ps, cudaStream_t stream) {
  int nblks = ps.rows;
  naive_kernel<<<nblks, TPB, sizeof(float) * ps.cols, stream>>>(ps);
  CUDA_CHECK(cudaPeekAtLastError());
}

}  // namespace fil
}  // namespace ML

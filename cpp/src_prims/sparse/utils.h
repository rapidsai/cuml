/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

namespace raft {
namespace sparse {

/**
 * Quantizes ncols to a valid blockdim, which is
 * a multiple of 32.
 *
 * @param[in] ncols number of blocks to quantize
 */
template <typename value_idx>
inline int block_dim(value_idx ncols) {
  int blockdim;
  if (ncols <= 32)
    blockdim = 32;
  else if (ncols <= 64)
    blockdim = 64;
  else if (ncols <= 128)
    blockdim = 128;
  else if (ncols <= 256)
    blockdim = 256;
  else if (ncols <= 512)
    blockdim = 512;
  else
    blockdim = 1024;

  return blockdim;
}

// add similar semantics for __match_any_sync pre-volta (SM_70)
#if __CUDA_ARCH__ < 700
/**
 * Returns a warp-level mask with 1's for all the threads
 * in the current warp that have the same key.
 * @tparam G
 * @param key
 * @return
 */
template <typename G>
__device__ __inline__ unsigned int __match_any_sync(unsigned int init_mask,
                                                    G key) {
  unsigned int mask = __ballot_sync(init_mask, true);
  unsigned int peer_group = 0;
  bool is_peer;

  do {
    // fetch key of first unclaimed lane and compare with this key
    is_peer = (key == __shfl_sync(mask, key, __ffs(mask) - 1));

    // determine which lanes had a match
    peer_group = __ballot_sync(mask, is_peer);

    // remove lanes with matching keys from the pool
    mask = mask ^ peer_group;

    // quit if we had a match
  } while (!is_peer);

  return peer_group;
}
#endif

__device__ __inline__ unsigned int get_lowest_peer(unsigned int peer_group) {
  return __ffs(peer_group) - 1;
}

template <typename value_idx>
__global__ void iota_fill_block_kernel(value_idx *indices, value_idx ncols) {
  int row = blockIdx.x;
  int tid = threadIdx.x;

  for (int i = tid; i < ncols; i += blockDim.x) {
    indices[row * ncols + i] = i;
  }
}

template <typename value_idx>
void iota_fill(value_idx *indices, value_idx nrows, value_idx ncols,
               cudaStream_t stream) {
  int blockdim = block_dim(ncols);

  iota_fill_block_kernel<<<nrows, blockdim, 0, stream>>>(indices, ncols);
}

template <typename T>
__device__ int get_stop_idx(T row, T m, T nnz, const T *ind) {
  int stop_idx = 0;
  if (row < (m - 1))
    stop_idx = ind[row + 1];
  else
    stop_idx = nnz;

  return stop_idx;
}

};  // namespace sparse
};  // namespace raft

/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

namespace MLCommon {
namespace Sparse {

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

/**
 * Returns a warp-level mask with 1's for all the threads
 * in the current warp that have the same key.
 * @tparam G
 * @param key
 * @return
 */
template <typename G>
__device__ __inline__ unsigned int get_peer_group(G key) {
  unsigned int peer_group = 0;
  bool is_peer;

  // in the beginning, all lanes are available
  unsigned int unclaimed = 0xffffffff;

  do {
    // fetch key of first unclaimed lane and compare with this key
    is_peer = (key == __shfl_sync(unclaimed, key, __ffs(unclaimed) - 1));

    // determine which lanes had a match
    peer_group = __ballot_sync(unclaimed, is_peer);

    // remove lanes with matching keys from the pool
    unclaimed = unclaimed ^ peer_group;

    // quit if we had a match
  } while (!is_peer);

  return peer_group;
}

__device__ __inline__ unsigned int get_lowest_peer(unsigned int peer_group) {
  return __ffs(peer_group) - 1;
}

};  // namespace Sparse
};  // namespace MLCommon

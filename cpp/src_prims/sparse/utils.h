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
};  // namespace Sparse
};  // namespace MLCommon

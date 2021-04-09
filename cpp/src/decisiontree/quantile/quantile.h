/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <memory>
#include <raft/mr/device/allocator.hpp>

template <class T, class L>
struct TemporaryMemory;
using deviceAllocator = raft::mr::device::allocator;

namespace ML {
namespace DecisionTree {

template <typename T, typename L>
void preprocess_quantile(const T *data, const unsigned int *rowids,
                         const int n_sampled_rows, const int ncols,
                         const int rowoffset, const int nbins,
                         std::shared_ptr<TemporaryMemory<T, L>> tempmem);

template <typename T>
void computeQuantiles(T *quantiles, int n_bins, const T *data, int n_rows,
                      int n_cols,
                      const std::shared_ptr<deviceAllocator> device_allocator,
                      cudaStream_t stream);

}  // namespace DecisionTree
}  // namespace ML

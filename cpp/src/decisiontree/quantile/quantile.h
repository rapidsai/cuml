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

#pragma once
#include "../memory.h"
template <typename T>
void preprocess_quantile(
  const T *data, const unsigned int *rowids, int n_sampled_rows, int ncols,
  int rowoffset, int nbins, T *h_quantile, T *d_quantile, T *temp_data,
  std::shared_ptr<MLCommon::deviceAllocator> device_allocator,
  cudaStream_t stream);

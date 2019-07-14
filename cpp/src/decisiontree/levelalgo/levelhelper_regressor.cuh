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
#include "levelkernel_regressor.cuh"
template <typename T, typename F>
void initial_metric_regression(
  T *labels, unsigned int *sample_cnt, const int nrows,
  const int n_unique_labels, std::vector<T> &meanvec, T &initial_metric,
  std::shared_ptr<TemporaryMemory<T, T>> tempmem) {
  int blocks = MLCommon::ceildiv(nrows, 128);
  if (blocks > 65536) blocks = 65536;
  
}

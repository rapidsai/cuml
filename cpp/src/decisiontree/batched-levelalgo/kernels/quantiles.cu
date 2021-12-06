
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
#include "../quantiles.cuh"

namespace ML {
namespace DT {

template <typename T>
__global__ void computeQuantilesSorted(T* quantiles,
                                       const int n_bins,
                                       const T* sorted_data,
                                       const int length)
{
  int tid          = threadIdx.x + blockIdx.x * blockDim.x;
  double bin_width = static_cast<double>(length) / n_bins;
  int index        = int(round((tid + 1) * bin_width)) - 1;
  index            = min(max(0, index), length - 1);
  if (tid < n_bins) { quantiles[tid] = sorted_data[index]; }

  return;
}

// instantiation
template __global__ void computeQuantilesSorted<float>(float* quantiles,
                                                       const int n_bins,
                                                       const float* sorted_data,
                                                       const int length);
template __global__ void computeQuantilesSorted<double>(double* quantiles,
                                                        const int n_bins,
                                                        const double* sorted_data,
                                                        const int length);

}  // end namespace DT
}  // end namespace ML
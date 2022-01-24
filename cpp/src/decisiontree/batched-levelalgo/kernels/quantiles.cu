
/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
__global__ void computeQuantilesKernel(
  T* quantiles, int* n_bins_unique, const T* sorted_data, const int n_bins_max, const int n_rows)
{
  // extern __shared__ char smem[];
  // auto* smem_quantiles = (T*)smem;
  // __shared__ int n_unique_bins;
  // int col          = blockIdx.x;  // each col per block
  // int base         = col * n_rows;
  double bin_width = static_cast<double>(n_rows) / n_bins_max;

  for (int bin = threadIdx.x; bin < n_bins_max; bin += blockDim.x) {
    // get index by interpolation
    int idx        = int(round((bin + 1) * bin_width)) - 1;
    idx            = min(max(0, idx), n_rows - 1);
    quantiles[bin] = sorted_data[idx];
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    // make quantiles unique, in-place
    auto new_last = thrust::unique(thrust::device, quantiles, quantiles + n_bins_max);
    // get the unique count
    *n_bins_unique = new_last - quantiles;
  }

  __syncthreads();
  return;
}

// instantiation
template __global__ void computeQuantilesKernel<float>(float* quantiles,
                                                       int* n_bins_unique,
                                                       const float* sorted_data,
                                                       const int n_bins_max,
                                                       const int n_rows);
template __global__ void computeQuantilesKernel<double>(double* quantiles,
                                                        int* n_bins_unique,
                                                        const double* sorted_data,
                                                        const int n_bins_max,
                                                        const int n_rows);

}  // end namespace DT
}  // end namespace ML
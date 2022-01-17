
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
__global__ void computeQuantilesBatchSorted(
  T* quantiles, int* useful_nbins, const T* sorted_data, const int n_bins, const int n_rows)
{
  extern __shared__ char smem[];
  auto* feature_quantiles = (T*)smem;
  __shared__ int n_unique_bins;
  int col          = blockIdx.x;  // each col per block
  int base         = col * n_rows;
  double bin_width = static_cast<double>(n_rows) / n_bins;

  for (int bin = threadIdx.x; bin < n_bins; bin += blockDim.x) {
    int offset             = int(round((bin + 1) * bin_width)) - 1;
    offset                 = min(max(0, offset), n_rows - 1);
    feature_quantiles[bin] = sorted_data[base + offset];
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    auto new_last = thrust::unique(thrust::device, feature_quantiles, feature_quantiles + n_bins);
    useful_nbins[blockIdx.x] = n_unique_bins = new_last - feature_quantiles;
  }

  __syncthreads();

  for (int bin = threadIdx.x; bin < n_unique_bins; bin += blockDim.x) {
    quantiles[col * n_bins + bin] = feature_quantiles[bin];
  }

  return;
}

// instantiation
template __global__ void computeQuantilesBatchSorted<float>(float* quantiles,
                                                            int* useful_nbins,
                                                            const float* sorted_data,
                                                            const int n_bins,
                                                            const int length);
template __global__ void computeQuantilesBatchSorted<double>(double* quantiles,
                                                             int* useful_nbins,
                                                             const double* sorted_data,
                                                             const int n_bins,
                                                             const int length);

}  // end namespace DT
}  // end namespace ML
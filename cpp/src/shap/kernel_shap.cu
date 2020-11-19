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

#include <cuml/explainer/kernel_shap.hpp>

#include <curand.h>
#include <curand_kernel.h>

namespace ML {
namespace Explainer {

/*
* Kernel distrubutes exact part of the kernel shap dataset
* Each block scatters the data of a row of `observations` into the (number of rows of
* background) in `dataset`, based on the row of `X`.
* So, given:
* background = [[0, 1, 2],
                [3, 4, 5]]
* observation = [100, 101, 102]
* X = [[1, 0, 1],
*      [0, 1, 1]]
*
* dataset (output):
* [[100, 1, 102],
*  [100, 4, 102]
*  [0, 101, 102],
*  [3, 101, 102]]
*
*
*/
template <typename DataT, typename IdxT>
__global__ void exact_rows_kernel_sm(DataT* X, IdxT nrows_X, IdxT ncols,
                                     DataT* background, IdxT nrows_background,
                                     DataT* dataset, DataT* observation) {
  extern __shared__ int idx[];
  int i, j;

  if (threadIdx.x < nrows_background) {
    // the first thread of each block gets the row of X that the block will use
    // for the scatter.
    if (threadIdx.x == 0) {
      for (i = 0; i < ncols; i++) {
        idx[i] = (int)X[blockIdx.x * ncols + i];
      }
    }
    __syncthreads();

    // all the threads now scatter the row, based on background and new observation
    int row = blockIdx.x * nrows_background + threadIdx.x;
#pragma unroll
    for (i = row; i < row + nrows_background; i += blockDim.x) {
#pragma unroll
      for (j = 0; j < ncols; j++) {
        if (idx[j] == 0) {
          dataset[i * ncols + j] = background[(i % nrows_background) * ncols + j];
        } else {
          dataset[i * ncols + j] = observation[j];
        }
      }
    }
  }
}

/*
* Similar kernel as above, but uses no shared memory for the index, in case
* it cannot fir in the shared memory of the device.
*
*/

template <typename DataT, typename IdxT>
__global__ void exact_rows_kernel(DataT* X, IdxT nrows_X, IdxT ncols,
                                  DataT* background, IdxT nrows_background,
                                  DataT* dataset, DataT* observation) {
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int i, j;

#pragma unroll
  for (i = tid; i < nrows_background; i += blockDim.x) {
#pragma unroll
    for (j = 0; j < ncols; j++) {
      if (X[blockIdx.x + j] == 0) {
        dataset[i * ncols + j] = background[(i % nrows_background) * ncols + j];
      } else {
        dataset[i * ncols + j] = observation[j];
      }
    }
  }
}

/*
* Kernel distrubutes sampled part of the kernel shap dataset
* The first thread of each block calculates the sampling of `k` entries of `observation`
* to scatter into `dataset`. Afterwards each block scatters the data of a row of `X` into the (number of rows of
* background) in `dataset`.
* So, given:
* background = [[0, 1, 2, 3],
                [5, 6, 7, 8]]
* observation = [100, 101, 102, 103]
* nsamples = [3, 2]
*
* X (output)
*      [[1, 0, 1, 1],
*       [0, 1, 1, 0]]
*
* dataset (output):
* [[100, 1, 102, 103],
*  [100, 6, 102, 103]
*  [0, 101, 102, 3],
*  [5, 101, 102, 8]]
*
*
*/
template <typename DataT, typename IdxT>
__global__ void sampled_rows_kernel(IdxT* nsamples, DataT* X, IdxT nrows_X,
                                    IdxT ncols, DataT* background,
                                    IdxT nrows_background, DataT* dataset,
                                    DataT* observation, uint64_t seed) {
  extern __shared__ int smps[];
  int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int i, j, k_blk;

  // see what k this block will generate
  k_blk = nsamples[blockIdx.x];

  if (threadIdx.x < nrows_background) {
    if (threadIdx.x == 0) {
      // thread 0 of block generates samples, reducing number of rng calls
      // calling curand only 3 * k times.
      // Sampling algo from: Li, Kim-Hung. "Reservoir-sampling algorithms
      // of time complexity O (n (1+ log (N/n)))." ACM Transactions on Mathematical
      // Software (TOMS) 20.4 (1994): 481-493.
      float w;
      curandState_t state;
      for (i = 0; i < k_blk; i++) {
        smps[i] = i;
      }
      curand_init((unsigned long long)seed, (unsigned long long)tid, 0, &state);

      w = exp(log(curand_uniform(&state)) / k_blk);

      while (i < ncols) {
        i = i + floor(log(curand_uniform(&state)) / log(1 - w)) + 1;
        if (i <= ncols) {
          smps[(int)(curand_uniform(&state) * k_blk)] = i;
          w = w * exp(log(curand_uniform(&state)) / k_blk);
        }
      }

      // write samples to 1-0 matrix
      for (i = 0; i < k_blk; i++) {
        X[i] = smps[i];
      }
    }

    // all threads write background line to their line

#pragma unroll
    for (i = tid; i < nrows_background; i += blockDim.x) {
#pragma unroll
      for (j = 0; j < ncols; j++) {
        dataset[i * ncols + j] = background[(i % nrows_background) * ncols + j];
      }
    }

    __syncthreads();

    // all threads write observation[samples] into their entry
#pragma unroll
    for (i = tid; i < nrows_background; i += blockDim.x) {
#pragma unroll
      for (j = 0; j < k_blk; j++) {
        dataset[i * ncols + smps[i]] = observation[smps[j]];
      }
    }
  }
}

template <typename DataT, typename IdxT>
void kernel_dataset_impl(const raft::handle_t& handle, DataT* X, IdxT nrows_X,
                         IdxT ncols, DataT* background, IdxT nrows_background,
                         DataT* combinations, DataT* observation, int* nsamples,
                         int len_samples, int maxsample, uint64_t seed) {
  const auto& handle_impl = handle;
  cudaStream_t stream = handle_impl.get_stream();

  IdxT nblks;
  IdxT nthreads;

  // calculate how many threads per block we need in multiples of 32
  nthreads = std::min(int(nrows_background / 32 + 1) * 32, 512);

  // number of blocks for exact part of the dataset
  nblks = nrows_X - len_samples;

  cudaDeviceProp prop;
  prop = handle_impl.get_device_properties();

  if (ncols * sizeof(DataT) <= prop.sharedMemPerMultiprocessor) {
    // each block calculates the combinations of an entry in X
    // at least nrows_background threads per block, multiple of 32
    exact_rows_kernel_sm<<<nblks, nthreads, ncols * sizeof(DataT), stream>>>(
      X, nrows_X, ncols, background, nrows_background, combinations,
      observation);
  } else {
    exact_rows_kernel<<<nblks, nthreads, 0, stream>>>(
      X, nrows_X, ncols, background, nrows_background, combinations,
      observation);
  }

  CUDA_CHECK(cudaPeekAtLastError());

  // check if random part of the dataset  is needed
  if (len_samples > 0) {
    // each block does a sample
    nblks = len_samples;

    // shared memory shouldn't be a problem since k will be small
    // due to distribution of shapley kernel weights
    sampled_rows_kernel<<<nblks, nthreads, maxsample * sizeof(int), stream>>>(
      nsamples, &X[(nrows_X - len_samples) * ncols], len_samples, ncols,
      background, nrows_background, combinations, observation, seed);
  }

  CUDA_CHECK(cudaPeekAtLastError());
}

void kernel_dataset(const raft::handle_t& handle, float* X, int nrows_X,
                    int ncols, float* background, int nrows_background,
                    float* dataset, float* observation, int* nsamples,
                    int len_nsamples, int maxsample, uint64_t seed) {
  kernel_dataset_impl(handle, X, nrows_X, ncols, background, nrows_background,
                      dataset, observation, nsamples, len_nsamples, maxsample,
                      seed);
}

void kernel_dataset(const raft::handle_t& handle, double* X, int nrows_X,
                    int ncols, double* background, int nrows_background,
                    double* dataset, double* observation, int* nsamples,
                    int len_nsamples, int maxsample, uint64_t seed) {
  kernel_dataset_impl(handle, X, nrows_X, ncols, background, nrows_background,
                      dataset, observation, nsamples, len_nsamples, maxsample,
                      seed);
}

}  // namespace Explainer
}  // namespace ML

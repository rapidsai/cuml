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

#include <math.h>
#include <raft/cudart_utils.h>
#include <common/device_buffer.hpp>
#include <cub/cub.cuh>
#include <cuml/common/cuml_allocator.hpp>
#include <raft/cuda_utils.cuh>

namespace MLCommon {

namespace Metrics {

/**
 * @brief kernel to calculate the precision score (currently only supporting binary class precision score)
 * @param gold: 1D array-like of ground truth class labels
 * @param pred: 1D array-like of predicted class labels
 * @param size: the size of array a and b
 * @param d_TP: pointer to the device memory that stores the aggregate true positive count
 * @param d_FP: pointer to the device memory that stores the aggregate false positive count
 */
template <int BLOCK_DIM_X>
__global__ void precision_score_kernel(const int *gold, const int *pred,
																			 int size, int *d_TP, int *d_FP) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = gridDim.x * blockDim.x;
  int localTP = 0;
  int localFP = 0;
  for (int i = idx; i < size; i += stride) {
  	if (pred[i] == 1) {
      if (gold[i] == 1) {
        localTP++;
      } else {
        localFP++;
      }
  	}
  }

  //specialize blockReduce for a 1D block
  typedef cub::BlockReduce<int, BLOCK_DIM_X> BlockReduce;

  //Allocate shared memory for blockReduce
  __shared__ typename BlockReduce::TempStorage temp_storage;

  //summing up thread-local counts specific to a block
  localTP = BlockReduce(temp_storage).Sum(localTP);
  localFP = BlockReduce(temp_storage).Sum(localFP);
  __syncthreads();


  //executed once per block
  if (threadIdx.x == 0) {
    raft::myAtomicAdd(d_TP, localTP);
    raft::myAtomicAdd(d_FP, localFP);
  }
}


double precision_score(const int* y, const int* y_hat, int size,
                       std::shared_ptr<MLCommon::deviceAllocator> allocator,
                       cudaStream_t stream) {
  // TODO: multi-class precision_score

  //creating device buffers needed for precision_score calculation
  //device variables
  MLCommon::device_buffer<int> d_gold(allocator, stream, size);
  MLCommon::device_buffer<int> d_pred(allocator, stream, size);
  MLCommon::device_buffer<int> d_TP(allocator, stream, 1);
  MLCommon::device_buffer<int> d_FP(allocator, stream, 1);

  //host variables
  int h_TP;
  int h_FP;

  //allocate and copy data from host memory to device memory
  CUDA_CHECK(cudaMemcpyAsync(d_gold.data(), y, size * sizeof(int),
             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_pred.data(), y_hat, size * sizeof(int),
             cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemsetAsync(d_TP.data(), 0, sizeof(int), stream));
  CUDA_CHECK(cudaMemsetAsync(d_FP.data(), 0, sizeof(int), stream));

  //kernel configuration
  static const int BLOCK_DIM_X = 16;
  dim3 numThreadsPerBlock(BLOCK_DIM_X);
  dim3 numBlocks(raft::ceildiv<int>(2, numThreadsPerBlock.x));

  //calling the kernel
  precision_score_kernel<BLOCK_DIM_X> 
    <<<numBlocks, numThreadsPerBlock, 0, stream>>>(
      d_gold.data(), d_pred.data(), size,
      d_TP.data(), d_FP.data());

  //updating in the host memory
  raft::update_host(&h_TP, d_TP.data(), 1, stream);
  raft::update_host(&h_FP, d_FP.data(), 1, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  double precision = 0;
  if (h_TP + h_FP > 0) {
    precision = h_TP / (h_TP + h_FP);
  } 
  return precision;
}


}; // end namespace Metrics
}; // end namespace MLCommon

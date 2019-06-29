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

#include <cub/cub.cuh>
#include <memory>
#include "common/cuml_allocator.hpp"
#include "common/device_buffer.hpp"
#include "cuda_utils.h"
#include "linalg/eltwise.h"

namespace MLCommon {
namespace Metrics {

///@todo: ColsPerBlk has been tested only for 32!
template <typename DataT, typename IdxT, int TPB, int ColsPerBlk = 32>
__global__ void weightedMeanKernel(DataT *mu, const DataT *data,
                                   const IdxT *counts, IdxT D, IdxT N) {
  constexpr int RowsPerBlkPerIter = TPB / ColsPerBlk;
  IdxT thisColId = threadIdx.x % ColsPerBlk;
  IdxT thisRowId = threadIdx.x / ColsPerBlk;
  IdxT colId = thisColId + ((IdxT)blockIdx.y * ColsPerBlk);
  IdxT rowId = thisRowId + ((IdxT)blockIdx.x * RowsPerBlkPerIter);
  DataT thread_data = DataT(0);
  const IdxT stride = RowsPerBlkPerIter * gridDim.x;
  for (IdxT i = rowId; i < N; i += stride) {
    thread_data +=
      (colId < D) ? data[i * D + colId] * (DataT)counts[i] : DataT(0);
  }
  __shared__ DataT smu[ColsPerBlk];
  if (threadIdx.x < ColsPerBlk) smu[threadIdx.x] = DataT(0);
  __syncthreads();
  myAtomicAdd(smu + thisColId, thread_data);
  __syncthreads();
  if (threadIdx.x < ColsPerBlk) myAtomicAdd(mu + colId, smu[thisColId]);
}

template <typename DataT, typename IdxT, int TPB>
__global__ void dispersionKernel(DataT *result, const DataT *clusters,
                                 const IdxT *clusterSizes, const DataT *mu,
                                 IdxT dim, IdxT nClusters) {
  IdxT tid = threadIdx.x + blockIdx.x * blockDim.x;
  IdxT len = dim * nClusters;
  IdxT stride = blockDim.x * gridDim.x;
  DataT sum = DataT(0);
  for (; tid < len; tid += stride) {
    IdxT col = tid % dim;
    IdxT row = tid / dim;
    DataT diff = clusters[tid] - mu[col];
    sum += diff * diff * DataT(clusterSizes[row]);
  }
  typedef cub::BlockReduce<DataT, TPB> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  auto acc = BlockReduce(temp_storage).Sum(sum);
  if (threadIdx.x == 0) myAtomicAdd(result, acc);
}

/**
 * @brief Compute cluster dispersion metric. This is very useful for
 * automatically finding the 'k' (in kmeans) that improves this metric.
 * @tparam DataT data type
 * @tparam IdxT index type
 * @tparam TPB threads block for kernels launched
 * @param centroids the cluster centroids. This is assumed to be row-major
 *   and of dimension (nClusters x dim)
 * @param clusterSizes number of points in the dataset which belong to each
 *   cluster. This is of length nClusters
 * @param globalCentroid compute the global weighted centroid of all cluster
 *   centroids. This is of length dim. Pass a nullptr if this is not needed
 * @param nClusters number of clusters
 * @param nPoints number of points in the dataset
 * @param dim dataset dimensionality
 * @param allocator device allocator
 * @param stream cuda stream
 * @return the cluster dispersion value
 */
template <typename DataT, typename IdxT = int, int TPB = 256>
DataT dispersion(const DataT *centroids, const IdxT *clusterSizes,
                 DataT *globalCentroid, IdxT nClusters, IdxT nPoints, IdxT dim,
                 std::shared_ptr<deviceAllocator> allocator,
                 cudaStream_t stream) {
  static const int RowsPerThread = 4;
  static const int ColsPerBlk = 32;
  static const int RowsPerBlk = (TPB / ColsPerBlk) * RowsPerThread;
  dim3 grid(ceildiv(nPoints, (IdxT)RowsPerBlk), ceildiv(dim, (IdxT)ColsPerBlk));
  device_buffer<DataT> mean(allocator, stream);
  device_buffer<DataT> result(allocator, stream, 1);
  DataT *mu = globalCentroid;
  if (globalCentroid == nullptr) {
    mean.resize(dim, stream);
    mu = mean.data();
  }
  CUDA_CHECK(cudaMemsetAsync(mu, 0, sizeof(DataT) * dim, stream));
  CUDA_CHECK(cudaMemsetAsync(result.data(), 0, sizeof(DataT), stream));
  weightedMeanKernel<DataT, IdxT, TPB, ColsPerBlk>
    <<<grid, TPB, 0, stream>>>(mu, centroids, clusterSizes, dim, nClusters);
  CUDA_CHECK(cudaGetLastError());
  DataT ratio = DataT(1) / DataT(nPoints);
  LinAlg::scalarMultiply(mu, mu, ratio, dim, stream);
  // finally, compute the dispersion
  constexpr int ItemsPerThread = 4;
  int nblks = ceildiv<int>(dim * nClusters, TPB * ItemsPerThread);
  dispersionKernel<DataT, IdxT, TPB><<<nblks, TPB, 0, stream>>>(
    result.data(), centroids, clusterSizes, mu, dim, nClusters);
  CUDA_CHECK(cudaGetLastError());
  DataT h_result;
  updateHost(&h_result, result.data(), 1, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
  return sqrt(h_result);
}

}  // end namespace Metrics
}  // end namespace MLCommon

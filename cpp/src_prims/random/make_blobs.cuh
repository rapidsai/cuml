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

#include <common/cudart_utils.h>
#include <cuml/common/cuml_allocator.hpp>
#include <vector>
#include "common/device_buffer.hpp"
#include "permute.cuh"
#include "rng.cuh"
#include <linalg/unary_op.cuh>

namespace MLCommon {
namespace Random {

namespace {

// generate the labels first and shuffle them instead of shuffling the dataset
template <typename IdxT>
void generate_labels(IdxT* labels, IdxT n_rows, IdxT n_clusters, bool shuffle,
                     cudaStream_t stream) {
  // always keep 'a' to be coprime to n_rows
  IdxT a = rand() % n_rows;
  while (gcd(a, n_rows) != 1) a = (a + 1) % n_rows;
  IdxT b = rand() % n_rows;
  auto op = [a, b, n_rows, n_clusters, shuffle] __device__(IdxT* ptr, IdxT idx) {
    if (shuffle) {
      idx = IdxT((a * int64_t(idx)) + b) % n_rows;
    }
    *ptr = idx % n_clusters;
  };
  LinAlg::writeOnlyUnaryOp<IdxT, decltype(op), IdxT>(labels, n_rows, op,
                                                     stream);
}

template <typename DataT, typename IdxT>
void generate_data(DataT* out, const IdxT* labels, IdxT n_rows, IdxT n_cols,
                   IdxT n_clusters, cudaStream_t stream, bool row_major,
                   const DataT* centers, const DataT* cluster_std,
                   const DataT cluster_std_scalar, Rng& rng) {
  auto op = [n_rows, n_cols, labels, centers, cluster_std, cluster_std_scalar]
    __device__(DataT val, IdxT idx) {
    IdxT cid, center_id;
    if (row_major) {
      cid = idx / n_cols;
      auto fid = idx % n_cols;
      center_id = cid * n_cols + fid;
    } else {
      cid = idx % n_rows;
      auto fid = idx / n_rows;
      center_id = cid + fid * n_rows;
    }
    auto sigma = cluster_std == nullptr ? cluster_std_scalar : cluster_std[cid];
    auto mu = centers[center_id];
    constexpr auto twoPi = DataT(2.0) * DataT(3.141592654);
    constexpr auto minus2 = -DataT(2.0);
    auto R = mySqrt(minus2 * myLog(val));
    auto theta = twoPi * val;
    val = mySin(theta) * R * sigma + mu;
    return val;
  };
  rng.custom_distribution<DataT, DataT, IdxT>(out, n_rows * n_cols, op, stream);
}

}  // namespace

template <typename DataT, typename IdxT>
__global__ void gatherKernel(DataT* out, const DataT* in, const IdxT* perms,
                             IdxT len) {
  IdxT tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < len) out[tid] = in[perms[tid]];
}

/**
 * @brief GPU-equivalent of sklearn.datasets.make_blobs
 *
 * @tparam DataT output data type
 * @tparam IdxT  indexing arithmetic type
 *
 * @param[out] out                generated data [on device]
 *                                [dim = n_rows x n_cols]
 * @param[out] labels             labels for the generated data [on device]
 *                                [len = n_rows]
 * @param[in]  n_rows             number of rows in the generated data
 * @param[in]  n_cols             number of columns in the generated data
 * @param[in]  n_clusters         number of clusters (or classes) to generate
 * @param[in]  allocator          device allocator for temporary allocations
 * @param[in]  stream             cuda stream to schedule the work on
 * @param[in]  centers            centers of each of the cluster, pass a nullptr
 *                                if you need this also to be generated randomly
 *                                [on device] [dim = n_clusters x n_cols]
 * @param[in]  cluster_std        standard deviation of each cluster center,
 *                                pass a nullptr if this is to be read from the
 *                                `cluster_std_scalar`. [on device]
 *                                [len = n_clusters]
 * @param[in]  cluster_std_scalar if 'cluster_std' is nullptr, then use this as
 *                                the std-dev across all dimensions.
 * @param[in]  shuffle            shuffle the generated dataset and labels
 * @param[in]  center_box_min     min value of box from which to pick cluster
 *                                centers. Useful only if 'centers' is nullptr
 * @param[in]  center_box_max     max value of box from which to pick cluster
 *                                centers. Useful only if 'centers' is nullptr
 * @param[in]  seed               seed for the RNG
 * @param[in]  type               RNG type
 */
template <typename DataT, typename IdxT>
void make_blobs(DataT* out, IdxT* labels, IdxT n_rows, IdxT n_cols,
                IdxT n_clusters, std::shared_ptr<deviceAllocator> allocator,
                cudaStream_t stream, const DataT* centers = nullptr,
                const DataT* cluster_std = nullptr,
                const DataT cluster_std_scalar = (DataT)1.0,
                bool shuffle = true, DataT center_box_min = (DataT)-10.0,
                DataT center_box_max = (DataT)10.0, uint64_t seed = 0ULL,
                GeneratorType type = GenPhilox) {
  Rng r(seed, type);
  // use the right centers buffer for data generation
  device_buffer<DataT> rand_centers(allocator, stream);
  const DataT* _centers;
  if (centers == nullptr) {
    rand_centers.resize(n_clusters * n_cols, stream);
    r.uniform(rand_centers.data(), n_clusters * n_cols, center_box_min,
              center_box_max, stream);
    _centers = rand_centers.data();
  } else {
    _centers = centers;
  }
  // use the right output buffer
  device_buffer<DataT> tmp_out(allocator, stream);
  device_buffer<IdxT> perms(allocator, stream);
  device_buffer<IdxT> tmp_labels(allocator, stream);
  DataT* _out;
  IdxT* _labels;
  if (shuffle) {
    tmp_out.resize(n_rows * n_cols, stream);
    perms.resize(n_rows, stream);
    tmp_labels.resize(n_rows, stream);
    _out = tmp_out.data();
    _labels = tmp_labels.data();
  } else {
    _out = out;
    _labels = labels;
  }
  // get the std info transferred to host
  std::vector<DataT> h_cluster_std(n_clusters, cluster_std_scalar);
  if (cluster_std != nullptr) {
    updateHost(&(h_cluster_std[0]), cluster_std, n_clusters, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }
  // generate data points for each cluster (assume equal distribution)
  IdxT rows_per_cluster = ceildiv(n_rows, n_clusters);
  for (IdxT i = 0, row_id = 0; i < n_clusters;
       ++i, row_id += rows_per_cluster) {
    IdxT current_rows = std::min(rows_per_cluster, n_rows - row_id);
    if (current_rows > 0) {
      r.normalTable<DataT, IdxT>(_out + row_id * n_cols, current_rows, n_cols,
                                 _centers + i * n_cols, nullptr,
                                 h_cluster_std[i], stream);
      r.fill(_labels + row_id, current_rows, (IdxT)i, stream);
    }
  }
  // shuffle, if asked for
  ///@todo: currently using a poor quality shuffle for better perf!
  if (shuffle) {
    permute<DataT, IdxT, IdxT>(perms.data(), out, _out, n_cols, n_rows, true,
                               stream);
    constexpr long Nthreads = 256;
    IdxT nblks = ceildiv<IdxT>(n_rows, Nthreads);
    gatherKernel<<<nblks, Nthreads, 0, stream>>>(labels, _labels, perms.data(),
                                                 n_rows);
  }
}

}  // end namespace Random
}  // end namespace MLCommon

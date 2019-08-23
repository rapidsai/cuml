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

#include <vector>
#include "common/cuml_allocator.hpp"
#include "common/device_buffer.hpp"
#include "permute.h"
#include "rng.h"
#include "utils.h"

namespace MLCommon {
namespace Random {

template <typename DataT, typename IdxT>
__global__ void gatherKernel(DataT* out, const DataT* in, const IdxT* perms,
                             IdxT len) {
  IdxT tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < len) out[tid] = in[perms[tid]];
}

/**
 * @brief GPU-equivalent of sklearn.datasets.make_blobs as documented here:
 * https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
 * @tparam DataT output data type
 * @tparam IdxT indexing arithmetic type
 * @param out the generated data on device (dim = n_rows x n_cols) in row-major
 * layout
 * @param labels labels for the generated data on device (dim = n_rows x 1)
 * @param n_rows number of rows in the generated data
 * @param n_cols number of columns in the generated data
 * @param n_cluster number of clusters (or classes) to generate
 * @param allocator device allocator to help allocate temporary buffers
 * @param stream cuda stream to schedule the work on
 * @param centers centers of each of the cluster, pass a nullptr if you need
 * this also to be generated randomly (dim = n_clusters x n_cols). This is
 * expected to be on device
 * @param cluster_std standard deviation of each of the cluster center, pass a
 * nullptr if you need this to be read from 'cluster_std_scalar'.
 * (dim = n_clusters x 1) This is expected to be on device
 * @param cluster_std_scalar if 'cluster_std' is nullptr, then use this as the
 * standard deviation across all dimensions.
 * @param shuffle shuffle the generated dataset and labels
 * @param center_box_min min value of the box from which to pick the cluster
 * centers. Useful only if 'centers' is nullptr
 * @param center_box_max max value of the box from which to pick the cluster
 * centers. Useful only if 'centers' is nullptr
 * @param seed seed for the RNG
 * @param type dataset generator type
 */
template <typename DataT, typename IdxT>
void make_blobs(DataT* out, int* labels, IdxT n_rows, IdxT n_cols,
                IdxT n_clusters, std::shared_ptr<deviceAllocator> allocator,
                cudaStream_t stream, const DataT* centers = nullptr,
                const DataT* cluster_std = nullptr,
                const DataT cluster_std_scalar = (DataT)1.0,
                bool shuffle = true, DataT center_box_min = (DataT)10.0,
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
  device_buffer<int> tmp_labels(allocator, stream);
  DataT* _out;
  int* _labels;
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
      r.fill(_labels + row_id, current_rows, (int)i, stream);
    }
  }
  // shuffle, if asked for
  ///@todo: currently using a poor quality shuffle for better perf!
  if (shuffle) {
    permute(perms.data(), out, _out, n_cols, n_rows, true, stream);
    constexpr int Nthreads = 256;
    int nblks = ceildiv<int>(n_rows, Nthreads);
    gatherKernel<<<nblks, Nthreads, 0, stream>>>(labels, _labels, perms.data(),
                                                 n_rows);
  }
}

}  // end namespace Random
}  // end namespace MLCommon

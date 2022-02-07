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

#pragma once

#include "permute.cuh"
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/linalg/unary_op.cuh>
#include <raft/random/rng.hpp>
#include <rmm/device_uvector.hpp>
#include <vector>

namespace MLCommon {
namespace Random {

namespace {

// generate the labels first and shuffle them instead of shuffling the dataset
template <typename IdxT>
void generate_labels(IdxT* labels,
                     IdxT n_rows,
                     IdxT n_clusters,
                     bool shuffle,
                     raft::random::Rng& r,
                     cudaStream_t stream)
{
  IdxT a, b;
  r.affine_transform_params(n_clusters, a, b);
  auto op = [=] __device__(IdxT * ptr, IdxT idx) {
    if (shuffle) { idx = IdxT((a * int64_t(idx)) + b); }
    idx %= n_clusters;
    // in the unlikely case of n_clusters > n_rows, make sure that the writes
    // do not go out-of-bounds
    if (idx < n_rows) { *ptr = idx; }
  };
  raft::linalg::writeOnlyUnaryOp<IdxT, decltype(op), IdxT>(labels, n_rows, op, stream);
}

template <typename DataT, typename IdxT>
DI void get_mu_sigma(DataT& mu,
                     DataT& sigma,
                     IdxT idx,
                     const IdxT* labels,
                     bool row_major,
                     const DataT* centers,
                     const DataT* cluster_std,
                     DataT cluster_std_scalar,
                     IdxT n_rows,
                     IdxT n_cols,
                     IdxT n_clusters)
{
  IdxT cid, fid;
  if (row_major) {
    cid = idx / n_cols;
    fid = idx % n_cols;
  } else {
    cid = idx % n_rows;
    fid = idx / n_rows;
  }
  IdxT center_id;
  if (cid < n_rows) {
    center_id = labels[cid];
  } else {
    center_id = 0;
  }

  if (fid >= n_cols) { fid = 0; }

  if (row_major) {
    center_id = center_id * n_cols + fid;
  } else {
    center_id += fid * n_clusters;
  }
  sigma = cluster_std == nullptr ? cluster_std_scalar : cluster_std[cid];
  mu    = centers[center_id];
}

template <typename DataT, typename IdxT>
void generate_data(DataT* out,
                   const IdxT* labels,
                   IdxT n_rows,
                   IdxT n_cols,
                   IdxT n_clusters,
                   cudaStream_t stream,
                   bool row_major,
                   const DataT* centers,
                   const DataT* cluster_std,
                   const DataT cluster_std_scalar,
                   raft::random::Rng& rng)
{
  auto op = [=] __device__(DataT & val1, DataT & val2, IdxT idx1, IdxT idx2) {
    DataT mu1, sigma1, mu2, sigma2;
    get_mu_sigma(mu1,
                 sigma1,
                 idx1,
                 labels,
                 row_major,
                 centers,
                 cluster_std,
                 cluster_std_scalar,
                 n_rows,
                 n_cols,
                 n_clusters);
    get_mu_sigma(mu2,
                 sigma2,
                 idx2,
                 labels,
                 row_major,
                 centers,
                 cluster_std,
                 cluster_std_scalar,
                 n_rows,
                 n_cols,
                 n_clusters);
    raft::random::box_muller_transform<DataT>(val1, val2, sigma1, mu1, sigma2, mu2);
  };
  rng.custom_distribution2<DataT, DataT, IdxT>(out, n_rows * n_cols, op, stream);
}

}  // namespace

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
 * @param[in]  stream             cuda stream to schedule the work on
 * @param[in]  row_major          whether input `centers` and output `out`
 *                                buffers are to be stored in row or column
 *                                major layout
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
void make_blobs(DataT* out,
                IdxT* labels,
                IdxT n_rows,
                IdxT n_cols,
                IdxT n_clusters,
                cudaStream_t stream,
                bool row_major                   = true,
                const DataT* centers             = nullptr,
                const DataT* cluster_std         = nullptr,
                const DataT cluster_std_scalar   = (DataT)1.0,
                bool shuffle                     = true,
                DataT center_box_min             = (DataT)-10.0,
                DataT center_box_max             = (DataT)10.0,
                uint64_t seed                    = 0ULL,
                raft::random::GeneratorType type = raft::random::GenPhilox)
{
  raft::random::Rng r(seed, type);
  // use the right centers buffer for data generation
  rmm::device_uvector<DataT> rand_centers(0, stream);
  const DataT* _centers;
  if (centers == nullptr) {
    rand_centers.resize(n_clusters * n_cols, stream);
    r.uniform(rand_centers.data(), n_clusters * n_cols, center_box_min, center_box_max, stream);
    _centers = rand_centers.data();
  } else {
    _centers = centers;
  }
  generate_labels(labels, n_rows, n_clusters, shuffle, r, stream);
  generate_data(out,
                labels,
                n_rows,
                n_cols,
                n_clusters,
                stream,
                row_major,
                _centers,
                cluster_std,
                cluster_std_scalar,
                r);
}

}  // end namespace Random
}  // end namespace MLCommon

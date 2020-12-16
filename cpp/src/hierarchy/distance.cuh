/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

#include <cuml/common/logger.hpp>

#include <cuml/cuml_api.h>
#include <raft/cudart_utils.h>
#include <raft/cuda_utils.cuh>

#include <sparse/coo.cuh>
#include <sparse/csr.cuh>

#include <common/cumlHandle.hpp>
#include <cuml/neighbors/knn.hpp>

#include <distance/distance.cuh>

#include <cuml/cluster/linkage.hpp>

#include <raft/linalg/distance_type.h>
#include <raft/mr/device/buffer.hpp>
#include <raft/sparse/mst/mst.cuh>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace ML {
namespace Linkage {
namespace Distance {

/**
 * Fills indices array of pairwise distance array
 * @tparam value_idx
 * @param indices
 * @param m
 */
template <typename value_idx>
__global__ void fill_indices(value_idx *indices, size_t m) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  value_idx v = tid / m;
  indices[tid] = v;
}

/**
 * Compute connected CSR of pairwise distances
 * @tparam value_idx
 * @tparam value_t
 * @param handle
 * @param X
 * @param m
 * @param n
 * @param metric
 * @param[out] indptr
 * @param[out] indices
 * @param[out] data
 */
template <typename value_idx, typename value_t>
void pairwise_distances(const raft::handle_t &handle, const value_t *X,
                        size_t m, size_t n, raft::distance::DistanceType metric,
                        value_idx *indptr, value_idx *indices, value_t *data) {
  auto d_alloc = handle.get_device_allocator();
  auto stream = handle.get_stream();

  size_t nnz = m * m;

  int blocks = raft::ceildiv(nnz, (size_t)1024);
  fill_indices<value_idx><<<blocks, 1024, 0, stream>>>(indices, m);

  thrust::device_ptr<value_idx> t_rows = thrust::device_pointer_cast(indptr);
  thrust::sequence(thrust::cuda::par.on(stream), indptr, indptr + m, 0, (int)m);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  value_idx v = m * m;  // TODO: No good.
  raft::update_device(indptr + m, &v, 1, stream);

  raft::mr::device::buffer<char> workspace(d_alloc, stream, 0);

  // @TODO: This is super expensive. Future versions need to eliminate
  //   the pairwise distance matrix, use KNN, or an MST based on the KNN graph
  MLCommon::Distance::pairwise_distance<value_t, size_t>(
    X, X, data, m, m, n, workspace, metric, stream);
}

template <typename value_idx>
value_idx build_k(value_idx n_samples, int c) {
  // from "kNN-MST-Agglomerative: A fast & scalable graph-based data clustering
  // approach on GPU"
  return floor(logf(n_samples)) + c;
}

// TODO: This will go away once KNN is using raft's distance type
ML::MetricType raft_distance_to_ml(raft::distance::DistanceType metric) {
  switch (metric) {
    case raft::distance::DistanceType::EucUnexpandedL1:
      return ML::METRIC_L1;

    case raft::distance::DistanceType::EucExpandedL2:
      return ML::METRIC_L2;

    case raft::distance::DistanceType::InnerProduct:
      return ML::METRIC_INNER_PRODUCT;

    default:
      throw raft::exception("Unsupported distance");
  }
}

template <typename in_t, typename out_t>
__global__ void conv_indices_kernel(in_t *inds, out_t *out) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  out_t v = inds[tid];
  out[tid] = v;
}

template <typename in_t, typename out_t, int tpb = 1024>
void conv_indices(in_t *inds, out_t *out, size_t size, cudaStream_t stream) {
  int blocks = raft::ceildiv(size, (size_t)tpb);
  conv_indices_kernel<<<blocks, tpb, 0, stream>>>(inds, out);
}

/**
 * Constructs a (symmetrized) knn graph
 * @tparam value_idx
 * @tparam value_t
 * @param handle
 * @param X
 * @param m
 * @param n
 * @param metric
 * @param indptr
 * @param indices
 * @param data
 * @param c
 */
template <typename value_idx = int64_t, typename value_t = float>
void knn_graph(const raft::handle_t &handle, const value_t *X, size_t m,
               size_t n, raft::distance::DistanceType metric,
               MLCommon::Sparse::COO<value_t, value_idx> &out, int c = 4) {
  int k = build_k(m, c);

  auto d_alloc = handle.get_device_allocator();
  auto stream = handle.get_stream();

  size_t nnz = m * k;

  raft::mr::device::buffer<value_idx> rows(d_alloc, stream, m * k);
  raft::mr::device::buffer<value_idx> indices(d_alloc, stream, m * k);
  raft::mr::device::buffer<value_t> data(d_alloc, stream, m * k);

  int blocks = raft::ceildiv(nnz, (size_t)1024);
  fill_indices<value_idx><<<blocks, 1024, 0, stream>>>(rows.data(), m);

  std::vector<value_t *> inputs;
  inputs.push_back(const_cast<value_t *>(X));

  std::vector<int> sizes;
  sizes.push_back(m);

  // This is temporary. Once faiss is updated, we should be able to
  // pass value_idx through to knn.
  raft::mr::device::buffer<int64_t> int64_indices(d_alloc, stream, m * k);

  ML::MetricType ml_metric = raft_distance_to_ml(metric);
  ML::brute_force_knn(handle, inputs, sizes, n, const_cast<value_t *>(X), m,
                      int64_indices.data(), data.data(), k, true, true,
                      ml_metric);

  conv_indices(int64_indices.data(), indices.data(), int64_indices.size(),
               stream);

  MLCommon::Sparse::coo_symmetrize<value_idx, value_t>(
    handle, rows.data(), indices.data(), data.data(), m, m, (size_t)(m * k),
    &out,
    [] __device__(value_idx row, value_idx col, value_t result,
                  value_t transpose) {
      // take max in the case of non-symmetric
      // metrics
      return fmaxf(result, transpose);
    });

  MLCommon::Sparse::COO<value_t> comp_coo(d_alloc, stream);
  MLCommon::Sparse::coo_remove_zeros<1024, value_t>(&out, &comp_coo, d_alloc,
                                                    stream);
}

};  // namespace Distance
};  // end namespace Linkage
};  // end namespace ML
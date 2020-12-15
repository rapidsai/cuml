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
#include <common/cumlHandle.hpp>

#include <raft/mr/device/buffer.hpp>
#include <raft/sparse/mst/mst.cuh>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

namespace ML {
namespace Linkage {
namespace MST {

/**
 * Fills indices array of pairwise distance array
 * @tparam value_idx
 * @param indices
 * @param m
 */
template<typename value_idx>
__global__ void fill_indices(value_idx *indices,
                             size_t m) {
  int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  value_idx v = tid / m;
  indices[tid] = v;
}


/**
 * Sorts a COO by its weight
 * @tparam value_idx
 * @tparam value_t
 * @param rows
 * @param cols
 * @param data
 * @param nnz
 * @param stream
 */
template <typename value_idx, typename value_t>
void sort_coo_by_data(value_idx *rows, value_idx *cols, value_t *data,
                      value_idx nnz, cudaStream_t stream) {

  thrust::device_ptr<value_idx> t_rows = thrust::device_pointer_cast(rows);
  thrust::device_ptr<value_idx> t_cols = thrust::device_pointer_cast(cols);
  thrust::device_ptr<value_t> t_data = thrust::device_pointer_cast(data);

  auto first = thrust::make_zip_iterator(thrust::make_tuple(t_rows, t_cols));

  thrust::sort_by_key(thrust::cuda::par.on(stream), t_data, t_data + nnz, first);
}

/**
 * Constructs an MST and sorts the resulting edges in ascending
 * order by their weight.
 * @tparam value_idx
 * @tparam value_t
 * @param[in] handle
 * @param[in] pw_dists
 * @param[in] m
 * @param[out] mst_src
 * @param[out] mst_dst
 * @param[out] mst_weight
 */
template <typename value_idx, typename value_t>
void build_sorted_mst(const raft::handle_t &handle,
                      const value_t *pw_dists,
                      size_t m,
                      value_idx *mst_src,
                      value_idx *mst_dst,
                      value_t *mst_weight) {

  auto d_alloc = handle.get_device_allocator();
  auto stream = handle.get_stream();

  raft::mr::device::buffer<value_idx> indptr(d_alloc, stream, m+1);
  raft::mr::device::buffer<value_idx> indices(d_alloc, stream, m * m);
  raft::mr::device::buffer<value_idx> color(d_alloc, stream, m * m);

  int blocks = raft::ceildiv((int)(m*m), 1024);
  fill_indices<value_idx><<<blocks, 1024, 0, stream>>>(indices.data(), m);

  thrust::device_ptr<value_idx> t_rows = thrust::device_pointer_cast(indptr.data());
  thrust::sequence(thrust::cuda::par.on(stream), indptr.data(), indptr.data()+m, 0, (int)m);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  value_idx v = m*m; // TODO: No good.
  raft::update_device(indptr.data()+m, &v, 1, stream);

  raft::print_device_vector("indptr: ", indptr.data(), m+1, std::cout);
  raft::print_device_vector("indices: ", indices.data(), m*2, std::cout);
  raft::print_device_vector("data: ", pw_dists, m, std::cout);

  CUML_LOG_INFO("Building MST");

  auto mst_coo =
    raft::mst::mst<value_idx, value_idx, value_t>(
      handle,
      indptr.data(),
      indices.data(),
      pw_dists,
      (value_idx)m,
      (value_idx)m * (value_idx)m,
      color.data(),
      stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  printf("n_edges: %d\n", mst_coo.n_edges);

  raft::print_device_vector("mst_src: ", mst_coo.src.data(), m-1, std::cout);
  raft::print_device_vector("mst_dst: ", mst_coo.dst.data(), m-1, std::cout);

  CUML_LOG_INFO("Sorting MST");

  sort_coo_by_data(mst_coo.src.data(), mst_coo.dst.data(), mst_coo.weights.data(),
                   mst_coo.n_edges, stream);

  // Would be nice if we could pass these directly into the MST
  raft::copy_async(mst_src, mst_coo.src.data(), mst_coo.n_edges, stream);
  raft::copy_async(mst_dst, mst_coo.dst.data(), mst_coo.n_edges, stream);
  raft::copy_async(mst_weight, mst_coo.weights.data(), mst_coo.n_edges, stream);

  CUML_LOG_INFO("DONE");
}

};  // end namespace MST
};  // end namespace Linkage
};  // end namespace ML
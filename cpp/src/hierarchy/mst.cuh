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
#include <common/cumlHandle.hpp>
#include <raft/cuda_utils.cuh>

#include <raft/mr/device/buffer.hpp>
#include <raft/sparse/mst/mst.cuh>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

namespace ML {
namespace Linkage {
namespace MST {

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

  thrust::sort_by_key(thrust::cuda::par.on(stream), t_data, t_data + nnz,
                      first);
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
void build_sorted_mst(const raft::handle_t &handle, const value_idx *indptr,
                      const value_idx *indices, const value_t *pw_dists,
                      size_t m, raft::mr::device::buffer<value_idx> &mst_src,
                      raft::mr::device::buffer<value_idx> &mst_dst,
                      raft::mr::device::buffer<value_t> &mst_weight) {
  auto d_alloc = handle.get_device_allocator();
  auto stream = handle.get_stream();

  raft::mr::device::buffer<value_idx> color(d_alloc, stream, m * m);

  auto mst_coo = raft::mst::mst<value_idx, value_idx, value_t>(
    handle, indptr, indices, pw_dists, (value_idx)m,
    (value_idx)m * (value_idx)m, color.data(), stream);

  sort_coo_by_data(mst_coo.src.data(), mst_coo.dst.data(),
                   mst_coo.weights.data(), mst_coo.n_edges, stream);

  // TODO: be nice if we could pass these directly into the MST
  mst_src.resize(mst_coo.n_edges, stream);
  mst_dst.resize(mst_coo.n_edges, stream);
  mst_weight.resize(mst_coo.n_edges, stream);

  raft::copy_async(mst_src.data(), mst_coo.src.data(), mst_coo.n_edges, stream);
  raft::copy_async(mst_dst.data(), mst_coo.dst.data(), mst_coo.n_edges, stream);
  raft::copy_async(mst_weight.data(), mst_coo.weights.data(), mst_coo.n_edges,
                   stream);
}

};  // end namespace MST
};  // end namespace Linkage
};  // end namespace ML
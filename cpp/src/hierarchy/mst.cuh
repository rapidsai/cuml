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

#include <cuml/cuml_api.h>
#include <raft/cudart_utils.h>
#include <common/cumlHandle.hpp>

#include <raft/mr/device/buffer.hpp>
#include <raft/sparse/mst/mst.cuh>

#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>

namespace ML {
namespace Linkage {
namespace MST {

template <typename value_idx, typename value_t>
void sort_coo_by_data(value_idx *rows, value_idx *cols, value_t *data,
                      value_idx nnz) {

  thrust::device_ptr<value_idx> t_rows = thrust::device_pointer_cast(rows);
  thrust::device_ptr<value_idx> t_cols = thrust::device_pointer_cast(cols);
  thrust::device_ptr<value_t> t_data = thrust::device_pointer_cast(data);

  auto first = thrust::make_zip_iterator(thrust::make_tuple(t_rows, t_cols));

  thrust::sort_by_key(t_data, t_data + nnz, first);
}

template <typename value_idx, typename value_t>
void build_sorted_mst(const raft::handle_t &handle,
                      const value_t *pw_dists,
                      size_t m,
                      value_idx *mst_src,
                      value_idx *mst_dst,
                      value_t *mst_weight) {
  auto d_alloc = handle.get_device_allocator();
  auto stream = handle.get_stream();

  raft::mr::device::buffer<value_idx> indptr(d_alloc, stream, m);
  raft::mr::device::buffer<value_idx> indices(d_alloc, stream, m * m);
  raft::mr::device::buffer<value_idx> color(d_alloc, stream, m * m);

  raft::Graph_COO<value_idx, value_idx, value_t> mst_coo =
    raft::mst::mst<value_idx, value_idx, value_t>(
      handle,
      indptr.data(),
      indices.data(),
      pw_dists,
      (value_idx)m,
      (value_idx)m * (value_idx)m,
      color.data(),
      stream);

  sort_coo_by_data(mst_coo.src.data(), mst_coo.dst.data(), mst_coo.weights.data(), mst_coo.n_edges);

  // Would be nice if we could pass these directly into the MST
  raft::copy_async(mst_src, mst_coo.src.data(), mst_coo.n_edges, stream);
  raft::copy_async(mst_dst, mst_coo.dst.data(), mst_coo.n_edges, stream);
  raft::copy_async(mst_weight, mst_coo.weights.data(), mst_coo.n_edges, stream);
}

};  // end namespace MST
};  // end namespace Linkage
};  // end namespace ML
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

#include <cuml/cuml_api.h>
#include <common/cumlHandle.hpp>

#include <raft/mr/device/buffer.hpp>
#include <raft/linalg/norm.cuh>

#include <raft/cudart_utils.h>
#include <distance/fused_l2_nn.cuh>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <cub/cub.cuh>



namespace raft {
namespace linkage {

template <typename value_idx, typename value_t>
struct FusedL2NNReduceOp {
  value_idx *colors;

  // output edge list cols
  value_idx *out_colors;

  FusedL2NNReduceOp(value_idx *colors_, value_idx *out_colors_): colors(colors_),
                                                                 out_colors(out_colors_){};

  typedef typename cub::KeyValuePair<value_idx, value_t> KVP;
  DI void operator()(value_idx rit, KVP *out, const KVP &other) {
    if (other.value < out->value && colors[rit] != colors[other.key]) {
      out->key = other.key;
      out->value = other.value;
    }
  }

  DI void operator()(value_idx rit, value_t *out, const KVP &other) {
    value_idx other_color = colors[other.key];
    if (other.value < *out && colors[rit] != other_color) {
      out_colors[rit] = other_color;
      *out = other.value;
    }
  }

  DI void init(value_t *out, value_t maxVal) { *out = maxVal; }
  DI void init(KVP *out, value_t maxVal) {
    out->key = -1;
    out->value = maxVal;
  }
};

/**
 * Connects components by computing a 1-nn to neighboring
 * components of each data point (e.g. component(nn) != component(self))
 * and reduces the results to include the smallest
 * @tparam value_idx
 * @tparam value_t
 * @param nn_indices
 * @param nn_dists
 * @param X
 * @param colors
 * @param n_rows
 * @param n_cols
 * @param d_alloc
 * @param stream
 */
template<typename value_idx, typename value_t>
void connect_components(value_idx *nn_indices,
                        value_t *nn_dists,
                        const value_t *X,
                        const value_idx *colors,
                        value_idx n_rows,
                        value_idx n_cols,
                        std::shared_ptr<MLCommon::deviceAllocator> &d_alloc,
                        cudaStream_t stream) {

  raft::mr::device::buffer<char> workspace(d_alloc, stream, n_rows);
  raft::mr::device::buffer<value_t> x_norm(d_alloc, stream, n_rows);

  raft::linalg::rowNorm(x_norm.data(), X, n_cols, n_rows,
                        raft::linalg::L2Norm, true, stream);

  raft::mr::device::buffer<value_idx> nn_colors(d_alloc, stream, n_rows);
  raft::mr::device::buffer<cub::KeyValuePair<value_idx, value_t>> temp_inds_dists(d_alloc, stream, n_rows);

  auto red_op = FusedL2NNReduceOp<value_idx, value_t>(colors, nn_colors);
  MLCommon::Distance::fusedL2NN<value_t, cub::KeyValuePair<value_idx, value_t>,
                                value_idx>(temp_inds_dists.data(), X, X,
                                           x_norm.data(), x_norm.data(),
                                n_rows, n_rows, n_cols, workspace.data(),
                                red_op, true, true, stream);

  // sort nn_colors by key w/ original colors
  thrust::device_ptr<value_idx> t_colors = thrust::device_pointer_cast(colors);
  thrust::device_ptr<value_idx> t_nn_colors = thrust::device_pointer_cast(nn_colors.data());
  thrust::device_ptr<value_t> t_data = thrust::device_pointer_cast(temp_inds_dists.data());

  auto first = thrust::make_zip_iterator(thrust::make_tuple(t_rows, t_cols));

  thrust::sort_by_key(thrust::cuda::par.on(stream), t_data, t_data + nnz,
                      first);
}




};  // end namespace linkage
};  // end namespace raft
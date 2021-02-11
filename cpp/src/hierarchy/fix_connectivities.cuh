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
#include <raft/sparse/convert/csr.cuh>
#include <sparse/coo.cuh>

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



template<typename value_idx, typename value_t, typename reduction>
__global__ void count_components_by_color(value_idx *out,
                                    const value_idx *colors_indptr,
                                    const value_idx *colors_nn,
                                    value_idx n_colors) {

  value_idx tid = blockDim.x * blockIdx.x + threadIdx.x;
  value_idx row = blockIdx.x;

  __shared__ extern value_idx smem[];

  value_idx start_offset = colors_indptr[row];
  value_idx stop_offset = colors_indptr[row+1];

  for(value_idx i = tid; i < (stop_offset - start_offset); i+= blockDim.x) {
    // todo: reduce by key
    smem[colors_nn] = 1;
  }

  value_idx unique = 0;
  for(value_idx i = tid; i < n_colors; i += blockDim.x) {
    unique += smem[i];
  }

  // todo: shfl reduce to 0 & write out
}

template<typename value_idx, typename value_t, typename reduction>
__global__ void min_components_by_color(value_idx *out_cols,
                                          value_idx *out_vals,
                                          value_idx *colors_indptr,
                                          value_idx *colors_nn,
                                          cub::KeyValuePair<value_idx, value_t> *idx_dists,
                                          value_idx n_colors) {

  value_idx tid = blockDim.x * blockIdx.x + threadIdx.x;
  value_idx row = blockIdx.x;

  __shared__ extern value_idx smem[];

  value_idx start_offset = colors_indptr[row];
  value_idx stop_offset = colors_indptr[row+1];

  for(value_idx i = tid; i < (stop_offset - start_offset); i+= blockDim.x) {
    // todo: reduce by key
    atomicMin(smem + colors_nn[i], idx_dists);
  }

  __syncthreads();

  for(value_idx i = tid; i < n_colors; i += blockDim.x) {
    // write out
  }
}


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
void connect_components(raft::sparse::COO<value_t, value_idx> &out,
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

  // sort nn_colors by key w/ original colors
  thrust::device_ptr<value_idx> t_colors = thrust::device_pointer_cast(colors);
  thrust::device_ptr<value_idx> t_nn_colors = thrust::device_pointer_cast(nn_colors.data());
  thrust::device_ptr<value_t> t_data = thrust::device_pointer_cast(temp_inds_dists.data());
  thrust::counting_iterator<value_idx> arg_sort(0);

  auto red_op = FusedL2NNReduceOp<value_idx, value_t>(colors, nn_colors);
  MLCommon::Distance::fusedL2NN<value_t, cub::KeyValuePair<value_idx, value_t>,
                                value_idx>(temp_inds_dists.data(), X, X,
                                           x_norm.data(), x_norm.data(),
                                n_rows, n_rows, n_cols, workspace.data(),
                                red_op, true, true, stream);

  // max_color + 1 = number of connected components
  value_idx n_components =
    *(thrust::max_element(thrust::cuda::par.on(stream), t_colors,
                          t_colors + n_rows)) + 1;

  raft::mr::device::buffer<value_idx> colors_indptr(d_alloc, stream, n_components);

  auto keys = thrust::make_zip_iterator(thrust::make_tuple(t_colors, t_nn_colors));
  auto vals = thrust::make_zip_iterator(thrust::make_tuple(t_data, arg_sort));

  // get all the colors in contiguous locations so we can map them to warps
  thrust::sort_by_key(thrust::cuda::par.on(stream), keys, keys + n_rows, vals);

  // create an indptr array for newly sorted colors
  raft::sparse::convert::sorted_coo_to_csr(colors, n_rows, colors_indptr.data(), n_components);

  // create degree array for closest components per row
  raft::mr::device::buffer<value_idx> color_neigh_degrees(d_alloc, stream, n_components);

  // map each component to a separate warp, perform warp reduce by key to find
  // number of unique components in output.
  count_components_by_color<<<n_components, 256, n_components * sizeof(value_idx), stream>>>(
    color_neigh_degrees.data(), colors_indptr.data(), temp_inds_dists.data(), n_components);

  // TODO: Sum color_neigh_degrees to find nnz

  // map each component to a separate warp, perform warp reduce by key to
  // find min for each component in output
  min_components_by_color<<<n_components, 256, n_components * sizeof(value_idx), stream>>>(
    out.);

  // symmetrize

  /**
   * void symmetrize(const raft::handle_t &handle, const value_idx *rows,
                const value_idx *cols, const value_t *vals, size_t m, size_t n,
                size_t nnz, raft::sparse::COO<value_t, value_idx> &out)
   */
   raft::sparse::linalg::symmetrize(
    
    );



}




};  // end namespace linkage
};  // end namespace raft
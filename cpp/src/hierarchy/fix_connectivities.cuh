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
#include <raft/sparse/coo.cuh>
#include <raft/sparse/linalg/symmetrize.cuh>

#include <raft/cudart_utils.h>
#include <distance/fused_l2_nn.cuh>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <limits>

#include <cub/cub.cuh>

namespace raft {
namespace linkage {

template <typename value_idx, typename value_t>
struct FixConnectivitiesRedOp {
  value_idx *colors;
  value_idx m;

  FixConnectivitiesRedOp(value_idx *colors_, value_idx m_): colors(colors_), m(m_){};

  typedef typename cub::KeyValuePair<value_idx, value_t> KVP;
  DI void operator()(value_idx rit, KVP *out, const KVP &other) {

    if (rit < m &&  other.value < out->value && colors[rit] != colors[other.key]) {
      out->key = other.key;
      out->value = other.value;
    }
  }

  DI KVP operator()(value_idx rit, const KVP& a, const KVP& b) {

    if(rit < m && a.value < b.value && colors[rit] != colors[a.key]) {
      return a;
    }
    else
      return b;
  }

  DI void init(value_t *out, value_t maxVal) { *out = maxVal; }
  DI void init(KVP *out, value_t maxVal) {
    out->key = -1;
    out->value = maxVal;
  }
};

template<typename value_idx>
__global__ void count_components_by_color_kernel(value_idx *out_indptr,
                                          const value_idx *colors_indptr,
                                          const value_idx *colors_nn,
                                          value_idx n_colors) {

  value_idx tid = blockDim.x * blockIdx.x + threadIdx.x;
  value_idx row = blockIdx.x;

  if(row >= n_colors) return;

  __shared__ extern value_idx count_smem[];

  value_idx start_offset = colors_indptr[row];
  value_idx stop_offset = colors_indptr[row+1];

  for(value_idx i = tid; i < (stop_offset - start_offset); i+= blockDim.x) {
    value_idx new_color = colors_nn[start_offset + i];
    count_smem[new_color] = 1;
  }

  __syncthreads();

  for(value_idx i = tid; i < n_colors; i += blockDim.x)
    atomicAdd(out_indptr+i, count_smem[i] > 0);
}

template<typename value_idx>
void count_components_by_color(value_idx *out_indptr,
                               const value_idx *colors_indptr,
                               const value_idx *colors_nn,
                               value_idx n_colors,
                               cudaStream_t stream) {

  count_components_by_color_kernel<<<n_colors, 256, n_colors * sizeof(value_idx), stream>>>(
    out_indptr, colors_indptr, colors_nn, n_colors);

}



/**
 * colors_nn is not assumed to be sorted wrt colors_indptr
 * so we need to perform atomic reductions in each thread.
 * @tparam value_idx
 * @tparam value_t
 * @tparam reduction
 * @param out_cols
 * @param out_vals
 * @param colors_indptr
 * @param colors_nn
 * @param idx_dists
 * @param n_colors
 */
template<typename value_idx, typename value_t>
__global__ void min_components_by_color_kernel(value_idx *out_cols,
                                        value_t *out_vals,
                                        value_idx *out_rows,
                                        const value_idx *out_indptr,
                                        const value_idx *colors_indptr,
                                        const value_idx *colors_nn,
                                        const value_idx *indices,
                                        const cub::KeyValuePair<value_idx, value_t> *kvp,
                                        value_idx n_colors) {

  __shared__ extern char min_smem[];

  int* mutex = (int*)min_smem;

  cub::KeyValuePair<value_idx, value_t> *min =
    (cub::KeyValuePair<value_idx, value_t>*)(mutex+n_colors);
  value_idx *src_inds = (value_idx*)(min+n_colors);

  value_idx *output_offset_i = (value_idx*)(src_inds+n_colors);

  if(threadIdx.x == 0) {
    output_offset_i[0] = 0;
  }

  value_idx start_offset = colors_indptr[blockIdx.x];
  value_idx stop_offset = colors_indptr[blockIdx.x+1];

  // initialize
  for(value_idx i = threadIdx.x; i < (stop_offset - start_offset); i+= blockDim.x) {
    auto skvp = min+i;
    skvp->key = -1;
    skvp->value = std::numeric_limits<value_t>::max();
  }

  __syncthreads();

  for(value_idx i = threadIdx.x; i < (stop_offset - start_offset); i+= blockDim.x) {
    value_idx new_color = colors_nn[start_offset + i];

    while (atomicCAS(mutex + new_color, 0, 1) == 1);
    __threadfence();
    auto cur_kvp = kvp[start_offset+i];
    if(cur_kvp.value < min[new_color].value) {
      src_inds[new_color] = indices[start_offset+i];
      min[new_color].key = cur_kvp.key;
      min[new_color].value = cur_kvp.value;
    }
    __threadfence();
    atomicCAS(mutex + new_color, 1, 0);
  }

  __syncthreads();

  value_idx out_offset = out_indptr[blockIdx.x];
  for(value_idx i = threadIdx.x; i < n_colors; i += blockDim.x) {
    auto min_color = min[i];
    if(min_color.key > -1) {
      __threadfence();

      value_idx cur_offset = output_offset_i[0];

      out_rows[out_offset+cur_offset] = src_inds[i];
      out_cols[out_offset+cur_offset] = min_color.key;
      out_vals[out_offset+cur_offset] = min_color.value;

      atomicAdd(output_offset_i, 1);
    }
  }
}

template<typename value_idx, typename value_t>
void min_components_by_color(raft::sparse::COO<value_t, value_idx> &coo,
                             const value_idx *out_indptr,
                             const value_idx *colors_indptr,
                             const value_idx *colors_nn,
                             const value_idx *indices,
                             const cub::KeyValuePair<value_idx, value_t> *kvp,
                             value_idx n_colors, cudaStream_t stream) {


  int smem_bytes = (n_colors * sizeof(int)) + (n_colors * sizeof(kvp)) + ((n_colors+1) * sizeof(value_idx));

  min_components_by_color_kernel<<<n_colors, 256, smem_bytes, stream>>>(
    coo.cols(), coo.vals(), coo.rows(), out_indptr, colors_indptr, colors_nn, indices, kvp, n_colors);
}

template<typename value_idx>
value_idx get_n_components(value_idx *colors, value_idx n_rows, cudaStream_t stream) {
  thrust::device_ptr<value_idx> t_colors = thrust::device_pointer_cast(colors);
  return *(thrust::max_element(thrust::cuda::par.on(stream), t_colors,
                          t_colors + n_rows)) + 1;
}


template<typename value_idx>
void build_output_colors_indptr(value_idx *degrees,
                                const value_idx *components_indptr,
                                const value_idx *nn_components,
                                value_idx n_components,
                                cudaStream_t stream) {

  CUDA_CHECK(cudaMemsetAsync(degrees, 0,
             (n_components+1) * sizeof(value_idx), stream));

  raft::print_device_vector("components_indptr", components_indptr, n_components+1, std::cout);
  raft::print_device_vector("nn_components", nn_components, 5, std::cout);

  /**
   * Create COO array by first computing CSR indptr w/ degrees of each
   * color followed by COO row/col/val arrays.
   */
  // map each component to a separate warp, perform warp reduce by key to find
  // number of unique components in output.

  CUML_LOG_DEBUG("Calling count_components_by_color");
  count_components_by_color(degrees, components_indptr,
                            nn_components, n_components, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUML_LOG_DEBUG("Performing exclusive scan");
  thrust::device_ptr<value_idx> t_degrees = thrust::device_pointer_cast(degrees);
  thrust::exclusive_scan(thrust::cuda::par.on(stream), t_degrees,
                         t_degrees + n_components + 1, t_degrees);

  CUML_LOG_DEBUG("Done.");
}

template<typename value_idx, typename value_t>
struct LookupColorOp {

  value_idx *colors;

  LookupColorOp(value_idx *colors_): colors(colors_) {}

  DI value_idx operator()(const cub::KeyValuePair<value_idx, value_t> &kvp) {
    return colors[kvp.key];
  }
};

template<typename value_idx, typename value_t>
void perform_1nn(cub::KeyValuePair<value_idx, value_t> *kvp,
                 value_idx *nn_colors,
                 value_idx *colors,
                 const value_t *X,
                 value_idx n_rows,
                 value_idx n_cols,
                 std::shared_ptr<raft::mr::device::allocator> d_alloc,
                 cudaStream_t stream) {


  raft::mr::device::buffer<int> workspace(d_alloc, stream, n_rows);
  raft::mr::device::buffer<value_t> x_norm(d_alloc, stream, n_rows);

  raft::linalg::rowNorm(x_norm.data(), X, n_cols, n_rows,
                        raft::linalg::L2Norm, true, stream);

  FixConnectivitiesRedOp<value_idx, value_t> red_op(colors, n_rows);
  MLCommon::Distance::fusedL2NN<value_t, cub::KeyValuePair<value_idx, value_t>,
    value_idx>(kvp, X, X,
               x_norm.data(), x_norm.data(),
               n_rows, n_rows, n_cols, workspace.data(),
               red_op, red_op, true, true, stream);

  thrust::device_ptr<cub::KeyValuePair<value_idx, value_t>> t_kvp = thrust::device_pointer_cast(kvp);
  thrust::device_ptr<value_idx> t_nn_colors = thrust::device_pointer_cast(nn_colors);

  LookupColorOp<value_idx, value_t> extract_colors_op(colors);
  thrust::transform(thrust::cuda::par.on(stream), t_kvp, t_kvp+n_rows, t_nn_colors, extract_colors_op);

  CUDA_CHECK(cudaStreamSynchronize(stream));


  raft::print_device_vector<value_idx>("nn_colors", nn_colors, n_rows, std::cout);

}

template<typename value_idx, typename value_t>
void sort_by_color(value_idx *colors,
                   value_idx *nn_colors,
                   cub::KeyValuePair<value_idx, value_t> *kvp,
                   value_idx *src_indices,
                   value_idx n_rows,
                   cudaStream_t stream) {
  thrust::device_ptr<value_idx> t_colors = thrust::device_pointer_cast(colors);
  thrust::device_ptr<value_idx> t_nn_colors = thrust::device_pointer_cast(nn_colors);
  thrust::device_ptr<cub::KeyValuePair<value_idx, value_t>> t_data = thrust::device_pointer_cast(kvp);
  thrust::device_ptr<value_idx> t_src_indices = thrust::device_pointer_cast(src_indices);

  thrust::counting_iterator<value_idx> arg_sort_iter(0);
  thrust::copy(arg_sort_iter, arg_sort_iter+n_rows, t_src_indices);

  auto keys = thrust::make_zip_iterator(thrust::make_tuple(t_colors));
  auto vals = thrust::make_zip_iterator(thrust::make_tuple(t_data, t_src_indices, t_nn_colors));

  // get all the colors in contiguous locations so we can map them to warps.
  thrust::sort_by_key(thrust::cuda::par.on(stream), keys, keys + n_rows, vals);
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
void connect_components(const raft::handle_t &handle,
                        raft::sparse::COO<value_t, value_idx> &out,
                        const value_t *X,
                        value_idx *colors,
                        value_idx n_rows,
                        value_idx n_cols) {

  auto d_alloc = handle.get_device_allocator();
  auto stream = handle.get_stream();

  value_idx n_components = get_n_components(colors, n_rows, stream);

  raft::print_device_vector("colors", colors, n_rows, std::cout);

  /**
   * First compute 1-nn for all colors where the color of each data point
   * is guaranteed to be != color of its nearest neighbor.
   */
  raft::mr::device::buffer<value_idx> nn_colors(d_alloc, stream, n_rows);
  raft::mr::device::buffer<cub::KeyValuePair<value_idx, value_t>> temp_inds_dists(d_alloc, stream, n_rows);
  raft::mr::device::buffer<value_idx> src_indices(d_alloc, stream, n_rows);
  raft::mr::device::buffer<value_idx> color_neigh_degrees(d_alloc, stream, n_components+1);
  raft::mr::device::buffer<value_idx> colors_indptr(d_alloc, stream, n_components);

  CUML_LOG_DEBUG("Performing 1nn");
  perform_1nn(temp_inds_dists.data(), nn_colors.data(), colors, X, n_rows, n_cols,
              d_alloc, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  /**
   * Sort data points by color (neighbors are not sorted)
   */
  // max_color + 1 = number of connected components
  // sort nn_colors by key w/ original colors

  CUML_LOG_DEBUG("Performing sort_by_color");
  sort_by_color(colors, nn_colors.data(), temp_inds_dists.data(), src_indices.data(),
                n_rows, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUML_LOG_DEBUG("Performing sorted_coo_to_csr");
  // create an indptr array for newly sorted colors
  raft::sparse::convert::sorted_coo_to_csr(colors, n_rows, colors_indptr.data(),
                                           n_components+1, d_alloc, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUML_LOG_DEBUG("Performing build_output_colors_indptr");
  // create output degree array for closest components per row
  build_output_colors_indptr(color_neigh_degrees.data(),
                             colors_indptr.data(),
                             nn_colors.data(),
                             n_components, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  raft::print_device_vector("color_neigh_degrees", color_neigh_degrees.data(), color_neigh_degrees.size(), std::cout);


  value_idx nnz;
  raft::update_host(&nnz, color_neigh_degrees.data()+n_components, 1, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  raft::sparse::COO<value_t, value_idx> min_edges(d_alloc, stream, nnz);

  CUML_LOG_DEBUG("Performing min_components_by_color");
  min_components_by_color(min_edges, color_neigh_degrees.data(),
                          colors_indptr.data(), nn_colors.data(), src_indices.data(),
                          temp_inds_dists.data(), n_components, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  raft::print_device_vector("min_edges", min_edges.rows(), nnz, std::cout);
  raft::print_device_vector("min_edges", min_edges.cols(), nnz, std::cout);
  raft::print_device_vector("min_edges", min_edges.vals(), nnz, std::cout);

  CUML_LOG_DEBUG("Performing symmetrize");
  // symmetrize
  raft::sparse::linalg::symmetrize(handle, min_edges.rows(),
                                 min_edges.cols(), min_edges.vals(),
                                 n_rows, n_rows, nnz, out);
  CUDA_CHECK(cudaStreamSynchronize(stream));

  CUML_LOG_DEBUG("Done.");
}




};  // end namespace linkage
};  // end namespace raft
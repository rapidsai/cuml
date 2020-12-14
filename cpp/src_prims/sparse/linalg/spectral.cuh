/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <raft/cudart_utils.h>

#include <raft/sparse/cusparse_wrappers.h>
#include <raft/cuda_utils.cuh>
#include <raft/mr/device/allocator.hpp>
#include <raft/mr/device/buffer.hpp>
#include <raft/spectral/partition.hpp>

#include <selection/knn.cuh>
#include <sparse/convert/csr.cuh>
#include <sparse/coo.cuh>

namespace raft {
namespace sparse {
namespace spectral {

template <typename T>
void fit_embedding(const raft::handle_t &handle,
                   int *rows, int *cols, T *vals,
                   int nnz, int n, int n_components, T *out) {

  auto stream = handle.get_stream();
  auto d_alloc = handle.get_device_allocator();
  raft::mr::device::buffer<int> src_offsets(d_alloc, stream, n + 1);
  raft::mr::device::buffer<int> dst_cols(d_alloc, stream, nnz);
  raft::mr::device::buffer<T> dst_vals(d_alloc, stream, nnz);
  convert::coo_to_csr(handle, rows, cols, vals, nnz, n, src_offsets.data(),
                      dst_cols.data(), dst_vals.data());

  raft::mr::device::buffer<T> eigVals(d_alloc, stream, n_components + 1);
  raft::mr::device::buffer<T> eigVecs(d_alloc, stream, n * (n_components + 1));
  raft::mr::device::buffer<int> labels(d_alloc, stream, n);

  CUDA_CHECK(cudaStreamSynchronize(stream));

  /**
   * Raft spectral clustering
   */
  using index_type = int;
  using value_type = T;

  index_type *ro = src_offsets.data();
  index_type *ci = dst_cols.data();
  value_type *vs = dst_vals.data();

  raft::matrix::sparse_matrix_t<index_type, value_type> const r_csr_m{
    handle, ro, ci, vs, n, nnz};

  index_type neigvs = n_components + 1;
  index_type maxiter = 4000;  //default reset value (when set to 0);
  value_type tol = 0.01;
  index_type restart_iter = 15 + neigvs;  //what cugraph is using
  auto t_exe_p = thrust::cuda::par.on(stream);
  using thrust_exe_policy_t = decltype(t_exe_p);

  raft::eigen_solver_config_t<index_type, value_type> cfg{neigvs, maxiter,
                                                          restart_iter, tol};

  raft::lanczos_solver_t<index_type, value_type> eig_solver{cfg};

  //cluster computation here is irrelevant,
  //hence define a no-op such solver to
  //feed partition():
  //
  struct no_op_cluster_solver_t {
    using index_type_t = index_type;
    using size_type_t = index_type;
    using value_type_t = value_type;

    std::pair<value_type_t, index_type_t> solve(
      handle_t const &handle, thrust_exe_policy_t t_exe_policy,
      size_type_t n_obs_vecs, size_type_t dim,
      value_type_t const *__restrict__ obs,
      index_type_t *__restrict__ codes) const {
      return std::make_pair<value_type_t, index_type_t>(0, 0);
    }
  };

  raft::spectral::partition(handle, t_exe_p, r_csr_m, eig_solver,
                            no_op_cluster_solver_t{}, labels.data(),
                            eigVals.data(), eigVecs.data());

  raft::copy<T>(out, eigVecs.data() + n, n * n_components, stream);

  CUDA_CHECK(cudaGetLastError());
}
};  // namespace spectral
};  // namespace sparse
};  // namespace raft

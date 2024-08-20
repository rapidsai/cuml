/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <cuml/solvers/lanczos.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/spectral/matrix_wrappers.hpp>

#include <raft/core/handle.hpp>

#include <raft/sparse/solver/lanczos.cuh>
#include <raft/sparse/solver/detail/lanczos.cuh>

#include <raft/sparse/convert/csr.cuh>
#include <raft/spectral/cluster_solvers.cuh>
#include <raft/spectral/eigen_solvers.cuh>
#include <raft/spectral/partition.cuh>

namespace ML {
namespace Solver {


template<typename index_type, typename value_type>
void lanczos_solver(
    const raft::handle_t& handle,
    index_type* rows,
    index_type* cols,
    value_type* vals,
    int nnz,
    int n,
    int n_components,
    int max_iterations,
    int ncv,
    value_type tolerance,
    uint64_t seed,
    value_type* v0,
    value_type* eigenvalues,
    value_type* eigenvectors
)
{
    auto stream = raft::resource::get_cuda_stream(handle);
    raft::device_vector_view<index_type, uint32_t, raft::row_major> rows_view = raft::make_device_vector_view<index_type, uint32_t, raft::row_major>(rows, n + 1);
    raft::device_vector_view<index_type, uint32_t, raft::row_major> cols_view = raft::make_device_vector_view<index_type, uint32_t, raft::row_major>(cols, nnz);
    raft::device_vector_view<value_type, uint32_t, raft::row_major> vals_view = raft::make_device_vector_view<value_type, uint32_t, raft::row_major>(vals, nnz);
    raft::device_vector_view<value_type, uint32_t, raft::row_major> v0_view = raft::make_device_vector_view<value_type, uint32_t, raft::row_major>(v0, n);
    raft::device_vector_view<value_type, uint32_t, raft::col_major> eigenvalues_view = raft::make_device_vector_view<value_type, uint32_t, raft::col_major>(eigenvalues, n_components);
    raft::device_matrix_view<value_type, uint32_t, raft::col_major> eigenvectors_view = raft::make_device_matrix_view<value_type, uint32_t, raft::col_major>(eigenvectors, n, n_components);

    raft::spectral::matrix::sparse_matrix_t<index_type, value_type> const csr_m{handle, rows_view.data_handle(), cols_view.data_handle(), vals_view.data_handle(), n, nnz};
    raft::sparse::solver::lanczos_solver_config<index_type, value_type> config{n_components, max_iterations, ncv, tolerance, seed};
    raft::sparse::solver::lanczos_compute_smallest_eigenvectors<index_type, value_type>(handle, csr_m, config, v0_view, eigenvalues_view, eigenvectors_view);
}

void lanczos_solver(
    const raft::handle_t& handle,
    int* rows,
    int* cols,
    double* vals,
    int nnz,
    int n,
    int n_components,
    int max_iterations,
    int ncv,
    double tolerance,
    uint64_t seed,
    double* v0,
    double* eigenvalues,
    double* eigenvectors
)
{
    lanczos_solver<int, double>(
        handle,
        rows,
        cols,
        vals,
        nnz,
        n,
        n_components,
        max_iterations,
        ncv,
        tolerance,
        seed,
        v0,
        eigenvalues,
        eigenvectors
    );
}


void lanczos_solver(
    const raft::handle_t& handle,
    int* rows,
    int* cols,
    float* vals,
    int nnz,
    int n,
    int n_components,
    int max_iterations,
    int ncv,
    float tolerance,
    uint64_t seed,
    float* v0,
    float* eigenvalues,
    float* eigenvectors
)
{
    lanczos_solver<int, float>(
        handle,
        rows,
        cols,
        vals,
        nnz,
        n,
        n_components,
        max_iterations,
        ncv,
        tolerance,
        seed,
        v0,
        eigenvalues,
        eigenvectors
    );
}



template<typename index_type, typename value_type>
void old_lanczos_solver(
    const raft::handle_t& handle,
    index_type* rows,
    index_type* cols,
    value_type* vals,
    int nnz,
    int n,
    int n_components,
    int max_iterations,
    int ncv,
    value_type tolerance,
    uint64_t seed,
    value_type* v0,
    value_type* eigenvalues,
    value_type* eigenvectors
)
{
    using T = value_type;
    auto stream = raft::resource::get_cuda_stream(handle);
    // rmm::device_uvector<int> src_offsets(n + 1, stream);
    // rmm::device_uvector<int> dst_cols(nnz, stream);
    // rmm::device_uvector<T> dst_vals(nnz, stream);
    // raft::sparse::convert::coo_to_csr(
    //     handle, rows, cols, vals, nnz, n, src_offsets.data(), dst_cols.data(), dst_vals.data());

    rmm::device_uvector<T> eigVals(n_components + 1, stream);
    rmm::device_uvector<T> eigVecs(n * (n_components + 1), stream);
    rmm::device_uvector<int> labels(n, stream);

    raft::resource::sync_stream(handle, stream);

    /**
    * Raft spectral clustering
    */
    // using index_type = int;
    // using value_type = T;

    // index_type* ro = src_offsets.data();
    // index_type* ci = dst_cols.data();
    // value_type* vs = dst_vals.data();

    index_type* ro = rows;
    index_type* ci = cols;
    value_type* vs = vals;

    raft::spectral::matrix::sparse_matrix_t<index_type, value_type> const r_csr_m{
        handle, ro, ci, vs, n, nnz};

    // index_type neigvs       = n_components + 1;
    // index_type maxiter      = 4000;  // default reset value (when set to 0);
    // value_type tol          = 0.01;
    // index_type restart_iter = 15 + neigvs;  // what cugraph is using

    index_type neigvs       = n_components + 1;
    index_type maxiter      = max_iterations;  // default reset value (when set to 0);
    value_type tol          = tolerance;
    index_type restart_iter = ncv;  // what cugraph is using

    raft::spectral::eigen_solver_config_t<index_type, value_type> cfg{
        neigvs, maxiter, restart_iter, tol};

    cfg.seed = seed;
    cfg.reorthogonalize = true;

    raft::spectral::lanczos_solver_t<index_type, value_type> eig_solver{cfg};

    // cluster computation here is irrelevant,
    // hence define a no-op such solver to
    // feed partition():
    //
    struct no_op_cluster_solver_t {
        using index_type_t = index_type;
        using size_type_t  = index_type;
        using value_type_t = value_type;

        std::pair<value_type_t, index_type_t> solve(raft::resources const& handle,
                                                    size_type_t n_obs_vecs,
                                                    size_type_t dim,
                                                    value_type_t const* __restrict__ obs,
                                                    index_type_t* __restrict__ codes) const
        {
        return std::make_pair<value_type_t, index_type_t>(0, 0);
        }
    };

    std::tuple<index_type, value_type, index_type> stats = raft::spectral::partition(handle,
                            r_csr_m,
                            eig_solver,
                            no_op_cluster_solver_t{},
                            labels.data(),
                            eigVals.data(),
                            eigVecs.data());
    
    std::cout << "old raft iters " << std::get<0>(stats) << std::endl;

    raft::copy<T>(eigenvectors, eigVecs.data() + n, n * n_components, stream);
    raft::copy<T>(eigenvalues, eigVals.data() + 1, n_components, stream);

    RAFT_CUDA_TRY(cudaGetLastError());
}


void old_lanczos_solver(
    const raft::handle_t& handle,
    int* rows,
    int* cols,
    float* vals,
    int nnz,
    int n,
    int n_components,
    int max_iterations,
    int ncv,
    float tolerance,
    uint64_t seed,
    float* v0,
    float* eigenvalues,
    float* eigenvectors
)
{
    old_lanczos_solver<int, float>(handle, rows, cols, vals, nnz, n, n_components, max_iterations, ncv, tolerance, seed, v0, eigenvalues, eigenvectors);
}


void old_lanczos_solver(
    const raft::handle_t& handle,
    int* rows,
    int* cols,
    double* vals,
    int nnz,
    int n,
    int n_components,
    int max_iterations,
    int ncv,
    double tolerance,
    uint64_t seed,
    double* v0,
    double* eigenvalues,
    double* eigenvectors
)
{
    old_lanczos_solver<int, double>(handle, rows, cols, vals, nnz, n, n_components, max_iterations, ncv, tolerance, seed, v0, eigenvalues, eigenvectors);
}


};  // namespace Solver
};  // end namespace ML

/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cuml/manifold/spectral_embedding.hpp>
#include <cuml/manifold/spectral_embedding_types.hpp>
#include <cuml/neighbors/knn.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/resources.hpp>
#include <raft/matrix/gather.cuh>
#include <raft/sparse/coo.hpp>
#include <raft/sparse/linalg/laplacian.cuh>
#include <raft/sparse/linalg/symmetrize.cuh>
#include <raft/sparse/op/filter.cuh>
#include <raft/sparse/solver/lanczos.cuh>
#include <raft/sparse/solver/lanczos_types.hpp>
#include <raft/util/cudart_utils.hpp>

#include <driver_types.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/remove.h>
#include <thrust/transform_scan.h>
#include <thrust/tuple.h>

#include <cuvs/neighbors/brute_force.hpp>

#include <cstdio>
#include <iostream>

// template <typename T>
// RAFT_KERNEL kernel_clamp_down_vector(T* vec, T*out, int k_neighbors, int size)
// {
//   int idx = threadIdx.x + blockIdx.x * blockDim.x;

//   // if (idx < size) { vec[idx] = (fabs(vec[idx]) < threshold) ? 0 : vec[idx]; }
// }

void scale_eigenvectors_by_diagonal(
  raft::device_matrix_view<float, int, raft::col_major> eigenvectors,
  const raft::device_vector<float, int>& diagonal,
  raft::resources const& handle)
{
  int n_rows = eigenvectors.extent(0);
  int n_cols = eigenvectors.extent(1);

  auto eigenvectors_ptr = eigenvectors.data_handle();
  auto diagonal_ptr     = diagonal.data_handle();

  auto policy = thrust::cuda::par.on(raft::resource::get_cuda_stream(handle));

  // For each row in the eigenvectors matrix
  thrust::for_each(policy,
                   thrust::counting_iterator<int>(0),
                   thrust::counting_iterator<int>(n_rows),
                   [eigenvectors_ptr, diagonal_ptr, n_cols, n_rows] __device__(int row) {
                     // For each column in this row
                     for (int col = 0; col < n_cols; col++) {
                       // With column-major layout, index is (row + col * n_rows)
                       int idx = row + col * n_rows;
                       // Divide by the corresponding diagonal element
                       eigenvectors_ptr[idx] /= diagonal_ptr[row];
                     }
                   });
}

template <typename T, typename IndexType>
void scale_csr_by_diagonal_symmetric(
  raft::device_csr_matrix_view<T, IndexType, IndexType, IndexType> csr_matrix,
  const raft::device_vector<T, IndexType>& diagonal,  // Vector of scaling factors
  raft::resources const& res)
{
  auto structure = csr_matrix.structure_view();
  auto nnz       = structure.get_nnz();

  auto values      = csr_matrix.get_elements().data();
  auto col_indices = structure.get_indices().data();
  auto row_offsets = structure.get_indptr().data();
  auto diag_ptr    = diagonal.data_handle();

  auto policy = thrust::cuda::par.on(raft::resource::get_cuda_stream(res));

  // For each row
  thrust::for_each(policy,
                   thrust::counting_iterator<IndexType>(0),
                   thrust::counting_iterator<IndexType>(structure.get_n_rows()),
                   [values, col_indices, row_offsets, diag_ptr] __device__(IndexType row) {
                     T row_scale = 1.0f / diag_ptr[row];  // Scale factor for this row

                     // For each element in this row
                     for (auto j = row_offsets[row]; j < row_offsets[row + 1]; j++) {
                       IndexType col = col_indices[j];
                       T col_scale   = 1.0f / diag_ptr[col];  // Scale factor for the column

                       // Scale by both row and column diagonal elements
                       values[j] = row_scale * values[j] * col_scale;
                     }
                   });
}

template <typename T, typename IndexType>
void scale_csr_by_diagonal_column_wise(
  raft::device_csr_matrix_view<T, IndexType, IndexType, IndexType> csr_matrix,
  const raft::device_vector<T, IndexType>& diagonal,  // Vector of scaling factors
  raft::resources const& res)
{
  auto structure = csr_matrix.structure_view();
  auto nnz       = structure.get_nnz();

  auto values      = csr_matrix.get_elements().data();
  auto col_indices = structure.get_indices().data();
  auto diag_ptr    = diagonal.data_handle();

  auto policy = thrust::cuda::par.on(raft::resource::get_cuda_stream(res));

  // Scale each element in CSR matrix by diagonal[col_index]
  thrust::for_each_n(policy,
                     thrust::counting_iterator<IndexType>(0),
                     nnz,
                     [values, col_indices, diag_ptr] __device__(IndexType idx) {
                       IndexType col = col_indices[idx];
                       values[idx] =
                         (1.0f / diag_ptr[col]) * values[idx] *
                         (1.0f / diag_ptr[col]);  // Multiply by diagonal element for this column
                     });
}

template <typename T, typename IndexType>
void set_csr_diagonal_to_ones_thrust(
  raft::device_csr_matrix_view<T, IndexType, IndexType, IndexType> csr_matrix,
  raft::resources const& res)
{
  auto structure = csr_matrix.structure_view();
  auto n_rows    = structure.get_n_rows();

  auto values      = csr_matrix.get_elements().data();
  auto col_indices = structure.get_indices().data();
  auto row_offsets = structure.get_indptr().data();

  auto policy = thrust::cuda::par.on(raft::resource::get_cuda_stream(res));

  thrust::for_each(policy,
                   thrust::counting_iterator<IndexType>(0),
                   thrust::counting_iterator<IndexType>(n_rows),
                   [values, col_indices, row_offsets] __device__(IndexType row) {
                     // For each row, find diagonal element (if it exists)
                     for (auto j = row_offsets[row]; j < row_offsets[row + 1]; j++) {
                       if (col_indices[j] == row) {
                         values[j] = static_cast<T>(1.0);
                         break;
                       }
                     }
                   });
}

template <typename T, typename IndexType>
void extract_csr_diagonal_thrust(
  raft::device_csr_matrix_view<T, IndexType, IndexType, IndexType> csr_matrix_view,
  raft::device_vector<T, IndexType>& diagonal,
  raft::resources const& res)
{
  auto structure = csr_matrix_view.structure_view();
  auto n_rows    = structure.get_n_rows();

  auto values      = csr_matrix_view.get_elements().data();
  auto col_indices = structure.get_indices().data();
  auto row_offsets = structure.get_indptr().data();
  auto diag_ptr    = diagonal.data_handle();

  auto policy = thrust::cuda::par.on(raft::resource::get_cuda_stream(res));

  thrust::for_each(policy,
                   thrust::counting_iterator<IndexType>(0),
                   thrust::counting_iterator<IndexType>(n_rows),
                   [values, col_indices, row_offsets, diag_ptr] __device__(IndexType row) {
                     // For each row, find diagonal element (if it exists)
                     for (auto j = row_offsets[row]; j < row_offsets[row + 1]; j++) {
                       if (col_indices[j] == row) {
                         diag_ptr[row] = values[j];
                         break;
                       }
                     }
                   });
}

// Functor to check if a value is odd
struct is_odd {
  __host__ __device__ bool operator()(const int x) { return x % 2 != 0; }
};

// Functor to check if both index and value are odd
struct both_index_and_value_odd {
  __host__ __device__ bool operator()(const thrust::tuple<int, int>& index_value_pair)
  {
    int index = thrust::get<0>(index_value_pair);
    int value = thrust::get<1>(index_value_pair);
    return (index % 2 != 0) && (value % 2 != 0);
  }
};

auto spectral_embedding_cuml(raft::resources const& handle,
                             raft::device_matrix_view<float, int, raft::row_major> nums,
                             raft::device_matrix_view<float, int, raft::col_major> embedding,
                             ML::spectral_embedding_config spectral_embedding_config) -> int
{
  // Define our sample data (similar to the Python example)
  const int n_samples     = nums.extent(0);
  const int n_features    = nums.extent(1);
  const int k             = spectral_embedding_config.n_neighbors;  // Number of neighbors
  const bool include_self = false;  // Set to false to exclude self-connections
  const bool drop_first   = spectral_embedding_config.drop_first;

  auto stream = raft::resource::get_cuda_stream(handle);
  // raft::device_resources res(stream);

  // If not including self, we need to request k+1 neighbors
  int k_search = include_self ? k : k + 1;

  cuvs::neighbors::brute_force::index_params index_params;
  index_params.metric = cuvs::distance::DistanceType::L2SqrtExpanded;

  auto d_indices   = raft::make_device_matrix<int64_t>(handle, n_samples, k_search);
  auto d_distances = raft::make_device_matrix<float>(handle, n_samples, k_search);

  auto index =
    cuvs::neighbors::brute_force::build(handle, index_params, raft::make_const_mdspan(nums));

  cuvs::neighbors::brute_force::search_params search_params;

  cuvs::neighbors::brute_force::search(
    handle, search_params, index, nums, d_indices.view(), d_distances.view());

  // Allocate memory for indices and distances
  // int64_t* d_indices;
  // float* d_distances;
  // cudaMalloc(&d_indices, n_samples * k_search * sizeof(int64_t));
  // cudaMalloc(&d_distances, n_samples * k_search * sizeof(float));

  // auto d_indices = raft::make_device_vector<int64_t>(handle, n_samples * k_search);
  // auto d_distances = raft::make_device_vector<float>(handle, n_samples * k_search);

  // // Setup input for brute_force_knn
  // std::vector<float*> input{nums.data_handle()};
  // std::vector<int> sizes{n_samples};

  // // auto myarr       = raft::make_device_vector<float>(handle, 2);

  // // Call brute_force_knn to find k_search nearest neighbors
  // ML::brute_force_knn(
  //   stream,
  //   input,
  //   sizes,
  //   n_features,
  //   nums.data_handle(),
  //   n_samples,
  //   d_indices.data_handle(),
  //   d_distances.data_handle(),
  //   k_search,
  //   true,
  //   true,
  //   cuvs::distance::DistanceType::L2SqrtExpanded,
  //   2.0f
  // );

  // Create iterators
  // auto index_iter = thrust::counting_iterator<int>(0);
  // auto value_iter = d_indices;

  // // Create a zip iterator that combines index and value
  // auto zip_begin = thrust::make_zip_iterator(thrust::make_tuple(index_iter, value_iter));
  // auto zip_end = thrust::make_zip_iterator(thrust::make_tuple(index_iter + n_samples * k_search,
  // value_iter + n_samples * k_search));

  // // Create functor to extract just the value from the tuple
  // struct get_value {
  //     __host__ __device__
  //     int operator()(const thrust::tuple<int, int>& index_value_pair) {
  //         return thrust::get<1>(index_value_pair);
  //     }
  // };

  // // Perform filtering and transformation in one step
  // auto new_end = thrust::transform_if(
  //   thrust::device,
  //   zip_begin,
  //   zip_end,
  //   d_indices,
  //   get_value(),               // Transform: extract the value
  //   both_index_and_value_odd() // Predicate: filter where both index and value are odd
  // );

  // auto mask = raft::make_device_vector<int64_t>(handle, n_samples * k_search);
  // auto prefixSum = raft::make_device_vector<int64_t>(handle, n_samples * k_search - n_samples);

  // Step 1: Create mask using Thrust transform

  // thrust::negate<int64_t> negate;
  // thrust::plus<int64_t> plus;

  // thrust::transform(thrust::device,
  //   d_indices,
  //   d_indices + n_samples * k_search,
  //   mask.data_handle(),
  //   negate
  // );

  // thrust::transform(
  //   thrust::device,
  //   thrust::make_zip_iterator(thrust::make_tuple(
  //       thrust::counting_iterator<int>(0),
  //       d_indices
  //   )),
  //   thrust::make_zip_iterator(thrust::make_tuple(
  //       thrust::counting_iterator<int>(0) + n_samples * k_search,
  //       d_indices + n_samples * k_search
  //   )),
  //   mask.data_handle(),
  //   [=] __device__ (const thrust::tuple<int, int>& t) {
  //       int idx = thrust::get<0>(t);
  //       int val = thrust::get<1>(t);
  //       return (val == (idx / k_search)) ? 0 : 1;
  //   }
  // );

  // thrust::copy_if(
  //   thrust::device,
  //   d_indices,
  //   d_indices + n_samples * k_search,
  //   mask.data_handle(),
  //   d_indices,
  //   [=] __device__ (int val) {
  //     return val;
  //   }
  // );

  // thrust::transform_exclusive_scan(
  //   thrust::device,
  //   thrust::make_zip_iterator(thrust::make_tuple(
  //       thrust::counting_iterator<int>(0),
  //       d_indices
  //   )),
  //   thrust::make_zip_iterator(thrust::make_tuple(
  //       thrust::counting_iterator<int>(0) + n_samples * k_search,
  //       d_indices + n_samples * k_search
  //   )),
  //   mask.data_handle(),
  //   [=] __device__ (const thrust::tuple<int, int>& t) {
  //       int idx = thrust::get<0>(t);
  //       int val = thrust::get<1>(t);
  //       return (val == (idx / k_search)) ? 0 : 1;
  //   },
  //   0,
  //   plus
  // );

  // raft::print_device_vector("d_indices", d_indices, n_samples * k_search, std::cout);
  // raft::print_device_vector("mask", mask.data_handle(), n_samples * k_search, std::cout);
  // raft::print_device_vector("prefixSum", prefixSum.data_handle(), n_samples * k_search -
  // n_samples, std::cout);

  // raft::print_device_vector("mask", mask.data_handle(), n_samples * k_search);
  // raft::print_device_vector("prefixSum", prefixSum.data_handle(), n_samples * k_search);

  // auto new_end = thrust::copy_if(thrust::device,
  //                                 d_indices,
  //                                 d_indices + n_samples * k_search,
  //                                 d_distances,
  //                                 is_odd());

  // Simple binarization: Set all distances to 1.0
  // float* h_ones = new float[n_samples * k_search];
  // for (int i = 0; i < n_samples * k_search; i++) {
  //   h_ones[i] = 1.0f;
  // }
  // cudaMemcpy(h_ones, d_distances, n_samples * k_search * sizeof(float), cudaMemcpyDeviceToHost);
  // cudaMemcpy(d_distances, h_ones, n_samples * k_search * sizeof(float), cudaMemcpyHostToDevice);

  // Get KNN indices for processing
  // int64_t* h_indices = new int64_t[n_samples * k_search];
  // cudaMemcpy(h_indices, d_indices, n_samples * k_search * sizeof(int64_t),
  // cudaMemcpyDeviceToHost); raft::print_device_vector("d_distances", d_distances.data_handle(),
  // n_samples * k_search, std::cout); raft::print_device_vector("d_indices",
  // d_indices.data_handle(), n_samples * k_search, std::cout);

  // Create a COO matrix for the KNN graph
  raft::sparse::COO<float> knn_coo(stream, n_samples, n_samples);

  // Resize COO to actual nnz
  size_t nnz = n_samples * k_search;
  knn_coo.allocate(nnz, n_samples, false, stream);

  auto knn_rows = raft::make_device_vector<int>(handle, nnz);
  auto knn_cols = raft::make_device_vector<int>(handle, nnz);

  thrust::transform(thrust::device,
                    d_indices.data_handle(),
                    d_indices.data_handle() + nnz,
                    knn_cols.data_handle(),
                    [] __device__(int64_t x) -> int { return static_cast<int>(x); });

  thrust::transform(thrust::device,
                    thrust::counting_iterator<int>(0),
                    thrust::counting_iterator<int>(nnz),
                    knn_rows.data_handle(),
                    [=] __device__(int64_t i) { return static_cast<int>(i / k_search); });

  // Copy COO data to device
  raft::copy(knn_coo.rows(), knn_rows.data_handle(), nnz, stream);
  raft::copy(knn_coo.cols(), knn_cols.data_handle(), nnz, stream);
  raft::copy(knn_coo.vals(), d_distances.data_handle(), nnz, stream);

  // printf("knn_coo.nnz = %ld\n", nnz);

  // raft::print_device_vector("knn_coo.rows", knn_coo.rows(), knn_coo.nnz, std::cout);
  // raft::print_device_vector("knn_coo.cols", knn_coo.cols(), knn_coo.nnz, std::cout);
  // raft::print_device_vector("knn_coo.vals", knn_coo.vals(), knn_coo.nnz, std::cout);//
  // knn_coo.nnz = nnz;

  raft::sparse::COO<float> coo_no_zeros(stream);  // Don't pre-allocate dimensions
  raft::sparse::op::coo_remove_zeros<float>(&knn_coo, &coo_no_zeros, stream);

  // binarize to 1s
  thrust::fill(thrust::device, coo_no_zeros.vals(), coo_no_zeros.vals() + coo_no_zeros.nnz, 1.0f);

  // printf("coo_no_zeros.nnz = %ld\n", coo_no_zeros.nnz);
  // raft::print_device_vector("coo_no_zeros.rows", coo_no_zeros.rows(), coo_no_zeros.nnz,
  // std::cout); raft::print_device_vector("coo_no_zeros.cols", coo_no_zeros.cols(),
  // coo_no_zeros.nnz, std::cout); raft::print_device_vector("coo_no_zeros.vals",
  // coo_no_zeros.vals(), coo_no_zeros.nnz, std::cout);

  // Create output COO for symmetrized result - create unallocated COO
  raft::sparse::COO<float> sym_coo1(stream);  // Don't pre-allocate dimensions

  // // Define the reduction function with the correct signature
  auto reduction_op = [] __device__(int row, int col, float a, float b) {
    // Only use the values, ignore row/col indices
    return 0.5f * (a + b);
  };

  // Symmetrize the matrix
  raft::sparse::linalg::coo_symmetrize(&coo_no_zeros, &sym_coo1, reduction_op, stream);

  //   raft::print_device_vector("sym_coo1.rows", sym_coo1.rows(), sym_coo1.nnz, std::cout);
  //   raft::print_device_vector("sym_coo1.cols", sym_coo1.cols(), sym_coo1.nnz, std::cout);
  //   raft::print_device_vector("sym_coo1.vals", sym_coo1.vals(), sym_coo1.nnz, std::cout);

  raft::sparse::op::coo_sort<float>(&sym_coo1, stream);

  // raft::print_device_vector("sym_coo1.rows", sym_coo1.rows(), sym_coo1.nnz, std::cout);
  // raft::print_device_vector("sym_coo1.cols", sym_coo1.cols(), sym_coo1.nnz, std::cout);
  // raft::print_device_vector("sym_coo1.vals", sym_coo1.vals(), sym_coo1.nnz, std::cout);

  raft::sparse::COO<float> sym_coo(stream);  // Don't pre-allocate dimensions
  raft::sparse::op::coo_remove_zeros<float>(&sym_coo1, &sym_coo, stream);

  // raft::print_device_vector("sym_coo.rows", sym_coo.rows(), sym_coo.nnz, std::cout);
  // raft::print_device_vector("sym_coo.cols", sym_coo.cols(), sym_coo.nnz, std::cout);
  // raft::print_device_vector("sym_coo.vals", sym_coo.vals(), sym_coo.nnz, std::cout);

  nnz = sym_coo.nnz;
  printf("\nSymmetrized COO Matrix (nnz=%ld):\n", nnz);

  using value_idx = int;
  using value_t   = float;
  using size_type = size_t;

  // Copy COO data to host for display
  // value_idx* h_rows = new value_idx[nnz];
  // value_idx* h_cols = new value_idx[nnz];
  // value_t* h_vals   = new value_t[nnz];

  // raft::copy(h_rows, sym_coo.rows(), nnz, stream);
  // raft::copy(h_cols, sym_coo.cols(), nnz, stream);
  // raft::copy(h_vals, sym_coo.vals(), nnz, stream);

  // // Print the COO elements
  // // for (size_type i = 0; i < nnz; i++) {
  // //   printf("(%d, %d) = %.1f\n", h_rows[i], h_cols[i], h_vals[i]);
  // // }

  // // For visualization, convert to dense matrix
  // float* h_dense_symmetric = new float[n_samples * n_samples]();
  // for (size_type i = 0; i < nnz; i++) {
  //   h_dense_symmetric[h_rows[i] * n_samples + h_cols[i]] = h_vals[i];
  // }

  // Print as dense matrix
  // printf("\nSymmetrized Dense Matrix:\n");
  // for (int i = 0; i < n_samples; i++) {
  //   for (int j = 0; j < n_samples; j++) {
  //     printf("%.1f ", h_dense_symmetric[i * n_samples + j]);
  //   }
  //   printf("\n");
  // }

  // Clean up
  // cudaFree(d_indices);
  // cudaFree(d_distances);

  // delete[] h_rows;
  // delete[] h_cols;
  // delete[] h_vals;
  // delete[] h_dense_symmetric;

  //   raft::print_device_vector("sym_coo.rows", sym_coo.rows(), sym_coo.nnz, std::cout);
  //   raft::print_device_vector("sym_coo.cols", sym_coo.cols(), sym_coo.nnz, std::cout);
  //   raft::print_device_vector("sym_coo.vals", sym_coo.vals(), sym_coo.nnz, std::cout);

  raft::sparse::op::coo_sort<float>(&sym_coo, stream);
  auto row_ind = raft::make_device_vector<int>(handle, sym_coo.n_rows + 1);
  raft::sparse::convert::sorted_coo_to_csr(&sym_coo, row_ind.data_handle(), stream);

  const int one = sym_coo.nnz;
  raft::copy(row_ind.data_handle() + row_ind.size() - 1, &one, 1, stream);

  // raft::print_device_vector("sym_coo.vals", sym_coo.vals(), sym_coo.nnz, std::cout);
  // raft::print_device_vector("sym_coo.cols", sym_coo.cols(), sym_coo.nnz, std::cout);
  // raft::print_device_vector("row_ind", row_ind.data_handle(), row_ind.size(), std::cout);

  auto csr_structure = raft::make_device_compressed_structure_view<int, int, int>(
    const_cast<int*>(row_ind.data_handle()),
    const_cast<int*>(sym_coo.cols()),
    sym_coo.n_rows,
    sym_coo.n_cols,
    sym_coo.nnz);

  auto csr_matrix_view = raft::make_device_csr_matrix_view<float, int, int, int>(
    const_cast<float*>(sym_coo.vals()), csr_structure);

  // L, dd = csgraph_laplacian(knn_graph_csr, normed=True, return_diag=True)
  // TODO: return diag and normed true
  auto laplacian           = raft::sparse::linalg::compute_graph_laplacian(handle, csr_matrix_view);
  auto laplacian_structure = laplacian.structure_view();

  // raft::print_device_vector("laplacian.get_elements().data()", laplacian.get_elements().data(),
  // laplacian_structure.get_nnz(), std::cout);
  // raft::print_device_vector("laplacian_structure.get_indices().data()",
  // laplacian_structure.get_indices().data(), laplacian_structure.get_nnz(), std::cout);
  // raft::print_device_vector("laplacian_structure.get_indptr().data()",
  // laplacian_structure.get_indptr().data(), laplacian_structure.get_n_rows() + 1, std::cout);

  auto diagonal = raft::make_device_vector<float>(handle, laplacian_structure.get_n_rows());
  extract_csr_diagonal_thrust(raft::make_device_csr_matrix_view<float, int, int, int>(
                                laplacian.get_elements().data(), laplacian_structure),
                              diagonal,
                              handle);

  thrust::transform(thrust::device,
                    diagonal.data_handle(),
                    diagonal.data_handle() + diagonal.size(),
                    diagonal.data_handle(),  // in-place
                    [] __device__(float x) { return std::sqrt(x); });

  // raft::print_device_vector("diagonal", diagonal.data_handle(), diagonal.size(), std::cout);

  // scale_csr_by_diagonal_column_wise(
  //     raft::make_device_csr_matrix_view<float, int, int, int>(
  //         laplacian.get_elements().data(),
  //         laplacian_structure
  //     ),
  //     diagonal,
  //     handle
  // );

  if (spectral_embedding_config.norm_laplacian) {
    scale_csr_by_diagonal_symmetric(raft::make_device_csr_matrix_view<float, int, int, int>(
                                      laplacian.get_elements().data(), laplacian_structure),
                                    diagonal,
                                    handle);
    set_csr_diagonal_to_ones_thrust(raft::make_device_csr_matrix_view<float, int, int, int>(
                                      laplacian.get_elements().data(), laplacian_structure),
                                    handle);
  }

  // extract_csr_diagonal_thrust(raft::make_device_csr_matrix_view<float, int, int,
  // int>(laplacian.get_elements().data(), laplacian_structure), diagonal, handle);
  // raft::print_device_vector("diagonal", diagonal.data_handle(), diagonal.size(), std::cout);

  // printf("laplacian_structure.get_indices().data() = %p\n",
  // laplacian_structure.get_indices().data());

  // L *= -1
  thrust::transform(thrust::device,
                    laplacian.get_elements().data(),
                    laplacian.get_elements().data() + laplacian_structure.get_nnz(),
                    laplacian.get_elements().data(),
                    [] __device__(float x) { return -x; });

  // raft::print_device_vector("laplacian.get_elements().data()", laplacian.get_elements().data(),
  // laplacian_structure.get_nnz(), std::cout);
  // raft::print_device_vector("laplacian_structure.get_indices().data()",
  // laplacian_structure.get_indices().data(), laplacian_structure.get_nnz(), std::cout);
  // raft::print_device_vector("laplacian_structure.get_indptr().data()",
  // laplacian_structure.get_indptr().data(), laplacian_structure.get_n_rows() + 1, std::cout);

  auto config           = raft::sparse::solver::lanczos_solver_config<float>();
  config.n_components   = spectral_embedding_config.n_components;
  config.max_iterations = 1000;
  config.ncv =
    std::min(laplacian_structure.get_n_rows(), std::max(2 * config.n_components + 1, 20));
  config.tolerance = 1e-5;
  config.which     = raft::sparse::solver::LANCZOS_WHICH::LA;
  config.seed      = spectral_embedding_config.seed;

  auto eigenvalues =
    raft::make_device_vector<float, int, raft::col_major>(handle, config.n_components);
  auto eigenvectors = raft::make_device_matrix<float, int, raft::col_major>(
    handle, laplacian_structure.get_n_rows(), config.n_components);

  // raft::sparse::solver::lanczos_compute_smallest_eigenvectors(handle, config,
  // laplacian.get_elements().data(), laplacian_structure.get_indices().data(),
  // laplacian_structure.get_indptr().data(), std::nullopt, eigenvalues, eigenvectors);

  raft::sparse::solver::lanczos_compute_smallest_eigenvectors<int, float>(
    handle,
    config,
    raft::make_device_csr_matrix_view<float, int, int, int>(laplacian.get_elements().data(),
                                                            laplacian_structure),
    std::nullopt,
    eigenvalues.view(),
    eigenvectors.view());

  raft::print_device_vector(
    "eigenvalues", eigenvalues.data_handle(), eigenvalues.size(), std::cout);
  // raft::print_device_vector(
  //   "eigenvectors", eigenvectors.data_handle(), eigenvectors.size(), std::cout);

  // raft::sparse::solver::lanczos_compute_smallest_eigenvectors(handle, config, 1, 1, 1, 1, 1, 1);
  // raft::sparse::solver::lanczos_compute_smallest_eigenvectors(handle, laplacian_structure,
  // laplacian, diagonal, eigenvalues, eigenvectors);

  if (spectral_embedding_config.norm_laplacian) {
    scale_eigenvectors_by_diagonal(eigenvectors.view(), diagonal, handle);
  }

  // Replace the direct copy with a gather operation that reverses columns

  // Create a sequence of reversed column indices
  config.n_components = drop_first ? config.n_components - 1 : config.n_components;
  auto col_indices    = raft::make_device_vector<int>(handle, config.n_components);
  thrust::sequence(thrust::device,
                   col_indices.data_handle(),
                   col_indices.data_handle() + config.n_components,
                   config.n_components - 1,  // Start from the last column index
                   -1                        // Decrement (move backward)
  );

  // Create row-major views of the column-major matrices
  // This is just a view re-interpretation, no data movement
  auto eigenvectors_row_view = raft::make_device_matrix_view<float, int, raft::row_major>(
    eigenvectors.data_handle(),
    eigenvectors.extent(1),  // Swap dimensions for the view
    eigenvectors.extent(0));

  auto embedding_row_view = raft::make_device_matrix_view<float, int, raft::row_major>(
    embedding.data_handle(),
    embedding.extent(1),  // Swap dimensions for the view
    embedding.extent(0));

  raft::matrix::gather<float, int, int>(
    handle,
    raft::make_const_mdspan(eigenvectors_row_view),  // Source matrix (as row-major view)
    raft::make_const_mdspan(col_indices.view()),     // Column indices to gather
    embedding_row_view                               // Destination matrix (as row-major view)
  );

  // copy eigenvectors to embedding
  // raft::copy(embedding.data_handle(), eigenvectors.data_handle(), eigenvectors.size(), stream);

  return 100;
}

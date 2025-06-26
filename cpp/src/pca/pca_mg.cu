/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include "pca.cuh"

#include <cuml/decomposition/pca.hpp>
#include <cuml/decomposition/pca_mg.hpp>
#include <cuml/decomposition/sign_flip_mg.hpp>

#include <cumlprims/opg/linalg/qr_based_svd.hpp>
#include <cumlprims/opg/matrix/matrix_utils.hpp>
#include <cumlprims/opg/stats/cov.hpp>
#include <cumlprims/opg/stats/mean.hpp>
#include <cumlprims/opg/stats/mean_center.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/matrix/math.cuh>
#include <raft/stats/mean_center.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cstddef>

using namespace MLCommon;

namespace ML {
namespace PCA {
namespace opg {

template <typename T>
void fit_impl(raft::handle_t& handle,
              std::vector<Matrix::Data<T>*>& input_data,
              Matrix::PartDescriptor& input_desc,
              T* components,
              T* explained_var,
              T* explained_var_ratio,
              T* singular_vals,
              T* mu,
              T* noise_vars,
              paramsPCAMG prms,
              cudaStream_t* streams,
              std::uint32_t n_streams,
              bool verbose)
{
  const auto& comm = handle.get_comms();

  Matrix::Data<T> mu_data{mu, prms.n_cols};

  Stats::opg::mean(handle, mu_data, input_data, input_desc, streams, n_streams);

  rmm::device_uvector<T> cov_data(prms.n_cols * prms.n_cols, streams[0]);
  auto cov_data_size = cov_data.size();
  Matrix::Data<T> cov{cov_data.data(), cov_data_size};

  Stats::opg::cov(handle, cov, input_data, input_desc, mu_data, true, streams, n_streams);

  ML::truncCompExpVars<T, mg_solver>(
    handle, cov.ptr, components, explained_var, explained_var_ratio, noise_vars, prms, streams[0]);

  T scalar = (prms.n_rows - 1);
  raft::matrix::seqRoot(explained_var, singular_vals, scalar, prms.n_components, streams[0], true);

  Stats::opg::mean_add(input_data, input_desc, mu_data, comm, streams, n_streams);
}

/**
 * @brief performs MNMG fit operation for the pca
 * @input param handle: the internal cuml handle object
 * @input param rank_sizes: includes all the partition size information for the rank
 * @input param input: input data
 * @input param components: principal components of the input data
 * @output param explained_var: explained var
 * @output param explained_var_ratio: the explained var ratio
 * @output param singular_vals: singular values of the data
 * @output param mu: mean of every column in input
 * @output param noise_vars: variance of the noise
 * @input param prms: data structure that includes all the parameters from input size to algorithm
 * @input param verbose
 */
template <typename T>
void fit_impl(raft::handle_t& handle,
              std::vector<Matrix::Data<T>*>& input_data,
              Matrix::PartDescriptor& input_desc,
              T* components,
              T* explained_var,
              T* explained_var_ratio,
              T* singular_vals,
              T* mu,
              T* noise_vars,
              paramsPCAMG prms,
              bool verbose)
{
  int rank = handle.get_comms().get_rank();

  // TODO: These streams should come from raft::handle_t
  // Reference issue https://github.com/rapidsai/cuml/issues/2470
  auto n_streams = input_desc.blocksOwnedBy(rank).size();
  cudaStream_t streams[n_streams];
  for (std::uint32_t i = 0; i < n_streams; i++) {
    RAFT_CUDA_TRY(cudaStreamCreate(&streams[i]));
  }

  if (prms.algorithm == mg_solver::COV_EIG_JACOBI || prms.algorithm == mg_solver::COV_EIG_DQ) {
    fit_impl(handle,
             input_data,
             input_desc,
             components,
             explained_var,
             explained_var_ratio,
             singular_vals,
             mu,
             noise_vars,
             prms,
             streams,
             n_streams,
             verbose);
  } else if (prms.algorithm == mg_solver::QR) {
    const raft::handle_t& h = handle;
    cudaStream_t stream     = h.get_stream();
    const auto& comm        = h.get_comms();

    // Center the data
    Matrix::Data<T> mu_data{mu, prms.n_cols};
    Stats::opg::mean(handle, mu_data, input_data, input_desc, streams, n_streams);
    Stats::opg::mean_center(input_data, input_desc, mu_data, comm, streams, n_streams);
    for (std::uint32_t i = 0; i < n_streams; i++) {
      handle.sync_stream(streams[i]);
    }

    // Allocate Q, S and V and call QR
    std::vector<Matrix::Data<T>*> uMatrixParts;
    Matrix::opg::allocate(h, uMatrixParts, input_desc, rank, stream);

    rmm::device_uvector<T> sVector(prms.n_cols, stream);

    rmm::device_uvector<T> vMatrix(prms.n_cols * prms.n_cols, stream);

    RAFT_CUDA_TRY(cudaMemset(vMatrix.data(), 0, prms.n_cols * prms.n_cols * sizeof(T)));

    LinAlg::opg::svdQR(h,
                       sVector.data(),
                       uMatrixParts,
                       vMatrix.data(),
                       true,
                       true,
                       prms.tol,
                       prms.n_iterations,
                       input_data,
                       input_desc,
                       rank);

    // sign flip
    sign_flip(handle, uMatrixParts, input_desc, vMatrix.data(), prms.n_cols, streams, n_streams);

    // Calculate instance variables
    rmm::device_uvector<T> explained_var_all(prms.n_cols, stream);
    rmm::device_uvector<T> explained_var_ratio_all(prms.n_cols, stream);

    T scalar = 1.0 / (prms.n_rows - 1);
    raft::matrix::power(sVector.data(), explained_var_all.data(), scalar, prms.n_cols, stream);
    raft::matrix::ratio(
      handle, explained_var_all.data(), explained_var_ratio_all.data(), prms.n_cols, stream);

    raft::matrix::truncZeroOrigin(
      sVector.data(), prms.n_cols, singular_vals, prms.n_components, std::size_t(1), stream);

    raft::matrix::truncZeroOrigin(explained_var_all.data(),
                                  prms.n_cols,
                                  explained_var,
                                  prms.n_components,
                                  std::size_t(1),
                                  stream);
    raft::matrix::truncZeroOrigin(explained_var_ratio_all.data(),
                                  prms.n_cols,
                                  explained_var_ratio,
                                  prms.n_components,
                                  std::size_t(1),
                                  stream);

    // Compute the scalar noise_vars defined as (pseudocode)
    // (n_components < min(n_cols, n_rows)) ? explained_var_all[n_components:].mean() : 0
    if (prms.n_components < prms.n_cols && prms.n_components < prms.n_rows) {
      raft::stats::mean<true>(noise_vars,
                              explained_var_all.data() + prms.n_components,
                              std::size_t{1},
                              prms.n_cols - prms.n_components,
                              false,
                              stream);
    } else {
      raft::matrix::setValue(noise_vars, noise_vars, T{0}, 1, stream);
    }

    raft::linalg::transpose(vMatrix.data(), prms.n_cols, stream);
    raft::matrix::truncZeroOrigin(
      vMatrix.data(), prms.n_cols, components, prms.n_components, prms.n_cols, stream);

    Matrix::opg::deallocate(h, uMatrixParts, input_desc, rank, stream);

    // Re-add mean to centered data
    Stats::opg::mean_add(input_data, input_desc, mu_data, comm, streams, n_streams);
  }

  for (std::uint32_t i = 0; i < n_streams; i++) {
    handle.sync_stream(streams[i]);
  }

  for (std::uint32_t i = 0; i < n_streams; i++) {
    RAFT_CUDA_TRY(cudaStreamDestroy(streams[i]));
  }
}

template <typename T>
void transform_impl(raft::handle_t& handle,
                    std::vector<Matrix::Data<T>*>& input,
                    const Matrix::PartDescriptor input_desc,
                    T* components,
                    std::vector<Matrix::Data<T>*>& trans_input,
                    T* singular_vals,
                    T* mu,
                    const paramsPCAMG prms,
                    cudaStream_t* streams,
                    std::uint32_t n_streams,
                    bool verbose)
{
  std::vector<Matrix::RankSizePair*> local_blocks = input_desc.partsToRanks;

  if (prms.whiten) {
    T scalar = T(sqrt(prms.n_rows - 1));
    raft::linalg::scalarMultiply(
      components, components, scalar, prms.n_cols * prms.n_components, streams[0]);
    raft::matrix::matrixVectorBinaryDivSkipZero<true, true>(
      components, singular_vals, prms.n_cols, prms.n_components, streams[0]);
  }

  for (std::size_t i = 0; i < input.size(); i++) {
    auto si = i % n_streams;

    raft::stats::meanCenter<false, true>(
      input[i]->ptr, input[i]->ptr, mu, prms.n_cols, local_blocks[i]->size, streams[si]);

    T alpha = T(1);
    T beta  = T(0);
    raft::linalg::gemm(handle,
                       input[i]->ptr,
                       local_blocks[i]->size,
                       prms.n_cols,
                       components,
                       trans_input[i]->ptr,
                       local_blocks[i]->size,
                       prms.n_components,
                       CUBLAS_OP_N,
                       CUBLAS_OP_T,
                       alpha,
                       beta,
                       streams[si]);

    raft::stats::meanAdd<false, true>(
      input[i]->ptr, input[i]->ptr, mu, prms.n_cols, local_blocks[i]->size, streams[si]);
  }

  if (prms.whiten) {
    raft::matrix::matrixVectorBinaryMultSkipZero<true, true>(
      components, singular_vals, prms.n_cols, prms.n_components, streams[0]);
    T scalar = T(1 / sqrt(prms.n_rows - 1));
    raft::linalg::scalarMultiply(
      components, components, scalar, prms.n_cols * prms.n_components, streams[0]);
  }

  for (std::uint32_t i = 0; i < n_streams; i++) {
    handle.sync_stream(streams[i]);
  }
}

/**
 * @brief performs MNMG transform operation for the pca.
 * @input param handle: the internal cuml handle object
 * @input param rank_sizes: includes all the partition size information for the rank
 * @input param n_parts: number of partitions
 * @input param input: input data
 * @input param components: principal components of the input data
 * @output param trans_input: transformed input data
 * @input param singular_vals: singular values of the data
 * @input param mu: mean of every column in input
 * @input param prms: data structure that includes all the parameters from input size to algorithm
 * @input param verbose
 */
template <typename T>
void transform_impl(raft::handle_t& handle,
                    Matrix::RankSizePair** rank_sizes,
                    std::uint32_t n_parts,
                    Matrix::Data<T>** input,
                    T* components,
                    Matrix::Data<T>** trans_input,
                    T* singular_vals,
                    T* mu,
                    paramsPCAMG prms,
                    bool verbose)
{
  // We want to update the API of this function, and other functions with
  // regards to https://github.com/rapidsai/cuml/issues/2471

  int rank = handle.get_comms().get_rank();

  std::vector<Matrix::RankSizePair*> ranksAndSizes(rank_sizes, rank_sizes + n_parts);
  std::vector<Matrix::Data<T>*> input_data(input, input + n_parts);
  Matrix::PartDescriptor input_desc(prms.n_rows, prms.n_cols, ranksAndSizes, rank);
  std::vector<Matrix::Data<T>*> trans_data(trans_input, trans_input + n_parts);

  // TODO: These streams should come from raft::handle_t
  auto n_streams = n_parts;
  cudaStream_t streams[n_streams];
  for (std::uint32_t i = 0; i < n_streams; i++) {
    RAFT_CUDA_TRY(cudaStreamCreate(&streams[i]));
  }

  transform_impl(handle,
                 input_data,
                 input_desc,
                 components,
                 trans_data,
                 singular_vals,
                 mu,
                 prms,
                 streams,
                 n_streams,
                 verbose);

  for (std::uint32_t i = 0; i < n_streams; i++) {
    handle.sync_stream(streams[i]);
  }

  for (std::uint32_t i = 0; i < n_streams; i++) {
    RAFT_CUDA_TRY(cudaStreamDestroy(streams[i]));
  }
}

template <typename T>
void inverse_transform_impl(raft::handle_t& handle,
                            std::vector<Matrix::Data<T>*>& trans_input,
                            Matrix::PartDescriptor trans_input_desc,
                            T* components,
                            std::vector<Matrix::Data<T>*>& input,
                            T* singular_vals,
                            T* mu,
                            paramsPCAMG prms,
                            cudaStream_t* streams,
                            std::uint32_t n_streams,
                            bool verbose)
{
  std::vector<Matrix::RankSizePair*> local_blocks = trans_input_desc.partsToRanks;

  if (prms.whiten) {
    T scalar = T(1 / sqrt(prms.n_rows - 1));
    raft::linalg::scalarMultiply(
      components, components, scalar, prms.n_rows * prms.n_components, streams[0]);
    raft::matrix::matrixVectorBinaryMultSkipZero<true, true>(
      components, singular_vals, prms.n_rows, prms.n_components, streams[0]);
  }

  for (std::size_t i = 0; i < local_blocks.size(); i++) {
    auto si = i % n_streams;
    T alpha = T(1);
    T beta  = T(0);

    raft::linalg::gemm(handle,
                       trans_input[i]->ptr,
                       local_blocks[i]->size,
                       prms.n_components,
                       components,
                       input[i]->ptr,
                       local_blocks[i]->size,
                       prms.n_cols,
                       CUBLAS_OP_N,
                       CUBLAS_OP_N,
                       alpha,
                       beta,
                       streams[si]);

    raft::stats::meanAdd<false, true>(
      input[i]->ptr, input[i]->ptr, mu, prms.n_cols, local_blocks[i]->size, streams[si]);
  }

  if (prms.whiten) {
    raft::matrix::matrixVectorBinaryDivSkipZero<true, true>(
      components, singular_vals, prms.n_rows, prms.n_components, streams[0]);
    T scalar = T(sqrt(prms.n_rows - 1));
    raft::linalg::scalarMultiply(
      components, components, scalar, prms.n_rows * prms.n_components, streams[0]);
  }

  for (std::uint32_t i = 0; i < n_streams; i++) {
    handle.sync_stream(streams[i]);
  }
}

/**
 * @brief performs MNMG inverse transform operation for the pca.
 * @input param handle: the internal cuml handle object
 * @input param rank_sizes: includes all the partition size information for the rank
 * @input param n_parts: number of partitions
 * @input param trans_input: transformed input data
 * @input param components: principal components of the input data
 * @output param input: input data
 * @input param singular_vals: singular values of the data
 * @input param mu: mean of every column in input
 * @input param prms: data structure that includes all the parameters from input size to algorithm
 * @input param verbose
 */
template <typename T>
void inverse_transform_impl(raft::handle_t& handle,
                            Matrix::RankSizePair** rank_sizes,
                            std::uint32_t n_parts,
                            Matrix::Data<T>** trans_input,
                            T* components,
                            Matrix::Data<T>** input,
                            T* singular_vals,
                            T* mu,
                            paramsPCAMG prms,
                            bool verbose)
{
  int rank = handle.get_comms().get_rank();

  std::vector<Matrix::RankSizePair*> ranksAndSizes(rank_sizes, rank_sizes + n_parts);
  Matrix::PartDescriptor trans_desc(prms.n_rows, prms.n_components, ranksAndSizes, rank);
  std::vector<Matrix::Data<T>*> trans_data(trans_input, trans_input + n_parts);

  std::vector<Matrix::Data<T>*> input_data(input, input + n_parts);

  // TODO: These streams should come from raft::handle_t
  auto n_streams = n_parts;
  cudaStream_t streams[n_streams];
  for (std::uint32_t i = 0; i < n_streams; i++) {
    RAFT_CUDA_TRY(cudaStreamCreate(&streams[i]));
  }

  inverse_transform_impl(handle,
                         trans_data,
                         trans_desc,
                         components,
                         input_data,
                         singular_vals,
                         mu,
                         prms,
                         streams,
                         n_streams,
                         verbose);

  for (std::uint32_t i = 0; i < n_streams; i++) {
    handle.sync_stream(streams[i]);
  }

  for (std::uint32_t i = 0; i < n_streams; i++) {
    RAFT_CUDA_TRY(cudaStreamDestroy(streams[i]));
  }
}

/**
 * @brief performs MNMG fit and transform operation for the pca.
 * @input param handle: the internal cuml handle object
 * @input param rank_sizes: includes all the partition size information for the rank
 * @input param n_parts: number of partitions
 * @input param input: input data
 * @output param trans_input: transformed input data
 * @output param components: principal components of the input data
 * @output param explained_var: explained var
 * @output param explained_var_ratio: the explained var ratio
 * @output param singular_vals: singular values of the data
 * @output param mu: mean of every column in input
 * @output param noise_vars: variance of the noise
 * @input param prms: data structure that includes all the parameters from input size to algorithm
 * @input param verbose
 */
template <typename T>
void fit_transform_impl(raft::handle_t& handle,
                        Matrix::RankSizePair** rank_sizes,
                        std::uint32_t n_parts,
                        Matrix::Data<T>** input,
                        Matrix::Data<T>** trans_input,
                        T* components,
                        T* explained_var,
                        T* explained_var_ratio,
                        T* singular_vals,
                        T* mu,
                        T* noise_vars,
                        paramsPCAMG prms,
                        bool verbose)
{
  int rank = handle.get_comms().get_rank();

  std::vector<Matrix::RankSizePair*> ranksAndSizes(rank_sizes, rank_sizes + n_parts);
  std::vector<Matrix::Data<T>*> input_data(input, input + n_parts);
  Matrix::PartDescriptor input_desc(prms.n_rows, prms.n_cols, ranksAndSizes, rank);
  std::vector<Matrix::Data<T>*> trans_data(trans_input, trans_input + n_parts);

  // TODO: These streams should come from raft::handle_t
  auto n_streams = n_parts;
  cudaStream_t streams[n_streams];
  for (std::uint32_t i = 0; i < n_streams; i++) {
    RAFT_CUDA_TRY(cudaStreamCreate(&streams[i]));
  }

  fit_impl(handle,
           input_data,
           input_desc,
           components,
           explained_var,
           explained_var_ratio,
           singular_vals,
           mu,
           noise_vars,
           prms,
           streams,
           n_streams,
           verbose);

  transform_impl(handle,
                 input_data,
                 input_desc,
                 components,
                 trans_data,
                 singular_vals,
                 mu,
                 prms,
                 streams,
                 n_streams,
                 verbose);

  sign_flip(handle, trans_data, input_desc, components, prms.n_components, streams, n_streams);

  for (std::uint32_t i = 0; i < n_streams; i++) {
    handle.sync_stream(streams[i]);
  }

  for (std::uint32_t i = 0; i < n_streams; i++) {
    RAFT_CUDA_TRY(cudaStreamDestroy(streams[i]));
  }
}

void fit(raft::handle_t& handle,
         std::vector<Matrix::Data<float>*>& input_data,
         Matrix::PartDescriptor& input_desc,
         float* components,
         float* explained_var,
         float* explained_var_ratio,
         float* singular_vals,
         float* mu,
         float* noise_vars,
         paramsPCAMG prms,
         bool verbose)
{
  fit_impl(handle,
           input_data,
           input_desc,
           components,
           explained_var,
           explained_var_ratio,
           singular_vals,
           mu,
           noise_vars,
           prms,
           verbose);
}

void fit(raft::handle_t& handle,
         std::vector<Matrix::Data<double>*>& input_data,
         Matrix::PartDescriptor& input_desc,
         double* components,
         double* explained_var,
         double* explained_var_ratio,
         double* singular_vals,
         double* mu,
         double* noise_vars,
         paramsPCAMG prms,
         bool verbose)
{
  fit_impl(handle,
           input_data,
           input_desc,
           components,
           explained_var,
           explained_var_ratio,
           singular_vals,
           mu,
           noise_vars,
           prms,
           verbose);
}

void fit_transform(raft::handle_t& handle,
                   Matrix::RankSizePair** rank_sizes,
                   std::uint32_t n_parts,
                   Matrix::floatData_t** input,
                   Matrix::floatData_t** trans_input,
                   float* components,
                   float* explained_var,
                   float* explained_var_ratio,
                   float* singular_vals,
                   float* mu,
                   float* noise_vars,
                   paramsPCAMG prms,
                   bool verbose)
{
  fit_transform_impl(handle,
                     rank_sizes,
                     n_parts,
                     input,
                     trans_input,
                     components,
                     explained_var,
                     explained_var_ratio,
                     singular_vals,
                     mu,
                     noise_vars,
                     prms,
                     verbose);
}

void fit_transform(raft::handle_t& handle,
                   Matrix::RankSizePair** rank_sizes,
                   std::uint32_t n_parts,
                   Matrix::doubleData_t** input,
                   Matrix::doubleData_t** trans_input,
                   double* components,
                   double* explained_var,
                   double* explained_var_ratio,
                   double* singular_vals,
                   double* mu,
                   double* noise_vars,
                   paramsPCAMG prms,
                   bool verbose)
{
  fit_transform_impl(handle,
                     rank_sizes,
                     n_parts,
                     input,
                     trans_input,
                     components,
                     explained_var,
                     explained_var_ratio,
                     singular_vals,
                     mu,
                     noise_vars,
                     prms,
                     verbose);
}

void transform(raft::handle_t& handle,
               Matrix::RankSizePair** rank_sizes,
               std::uint32_t n_parts,
               Matrix::Data<float>** input,
               float* components,
               Matrix::Data<float>** trans_input,
               float* singular_vals,
               float* mu,
               paramsPCAMG prms,
               bool verbose)
{
  transform_impl(
    handle, rank_sizes, n_parts, input, components, trans_input, singular_vals, mu, prms, verbose);
}

void transform(raft::handle_t& handle,
               Matrix::RankSizePair** rank_sizes,
               std::uint32_t n_parts,
               Matrix::Data<double>** input,
               double* components,
               Matrix::Data<double>** trans_input,
               double* singular_vals,
               double* mu,
               paramsPCAMG prms,
               bool verbose)
{
  transform_impl(
    handle, rank_sizes, n_parts, input, components, trans_input, singular_vals, mu, prms, verbose);
}

void inverse_transform(raft::handle_t& handle,
                       Matrix::RankSizePair** rank_sizes,
                       std::uint32_t n_parts,
                       Matrix::Data<float>** trans_input,
                       float* components,
                       Matrix::Data<float>** input,
                       float* singular_vals,
                       float* mu,
                       paramsPCAMG prms,
                       bool verbose)
{
  inverse_transform_impl(
    handle, rank_sizes, n_parts, trans_input, components, input, singular_vals, mu, prms, verbose);
}

void inverse_transform(raft::handle_t& handle,
                       Matrix::RankSizePair** rank_sizes,
                       std::uint32_t n_parts,
                       Matrix::Data<double>** trans_input,
                       double* components,
                       Matrix::Data<double>** input,
                       double* singular_vals,
                       double* mu,
                       paramsPCAMG prms,
                       bool verbose)
{
  inverse_transform_impl(
    handle, rank_sizes, n_parts, trans_input, components, input, singular_vals, mu, prms, verbose);
}

}  // namespace opg
}  // namespace PCA
}  // namespace ML

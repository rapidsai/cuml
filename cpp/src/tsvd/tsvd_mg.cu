/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <cuml/common/cuml_allocator.hpp>
#include <cuml/common/device_buffer.hpp>
#include <cuml/decomposition/sign_flip_mg.hpp>
#include <cuml/decomposition/tsvd.hpp>
#include <cuml/decomposition/tsvd_mg.hpp>
#include <opg/linalg/mm_aTa.hpp>
#include <opg/stats/mean.hpp>
#include <opg/stats/mean_center.hpp>
#include <opg/stats/stddev.hpp>
#include <raft/comms/comms.hpp>
#include <raft/cuda_utils.cuh>
#include <raft/linalg/eltwise.cuh>
#include <raft/matrix/math.cuh>
#include <raft/stats/mean_center.cuh>
#include "tsvd.cuh"

using namespace MLCommon;

namespace ML {
namespace TSVD {
namespace opg {

template <typename T>
void fit_impl(raft::handle_t &handle,
              std::vector<Matrix::Data<T> *> &input_data,
              Matrix::PartDescriptor &input_desc, T *components,
              T *singular_vals, paramsTSVD prms, cudaStream_t *streams,
              int n_streams, bool verbose) {
  const auto &comm = handle.get_comms();
  cublasHandle_t cublas_handle = handle.get_cublas_handle();
  const auto allocator = handle.get_device_allocator();

  // This variable should be updated to use `size_t`
  // Reference issue https://github.com/rapidsai/cuml/issues/2459
  int len = prms.n_cols * prms.n_cols;

  device_buffer<T> cov_data(allocator, streams[0], len);
  size_t cov_data_size = cov_data.size();
  Matrix::Data<T> cov{cov_data.data(), cov_data_size};

  LinAlg::opg::mm_aTa(handle, cov, input_data, input_desc, streams, n_streams);

  device_buffer<T> components_all(allocator, streams[0], len);
  device_buffer<T> explained_var_all(allocator, streams[0], prms.n_cols);

  ML::calEig(handle, cov.ptr, components_all.data(), explained_var_all.data(),
             prms, streams[0]);

  raft::matrix::truncZeroOrigin(components_all.data(), prms.n_cols, components,
                                prms.n_components, prms.n_cols, streams[0]);

  T scalar = T(1);
  raft::matrix::seqRoot(explained_var_all.data(), singular_vals, scalar,
                        prms.n_components, streams[0]);
}

/**
 * @brief performs MNMG fit operation for the tsvd
 * @input param handle: the internal cuml handle object
 * @input param rank_sizes: includes all the partition size information for the rank
 * @input param n_parts: number of partitions
 * @input param input: input data
 * @output param components: principal components of the input data
 * @output param singular_vals: singular values of the data
 * @input param prms: data structure that includes all the parameters from input size to algorithm
 * @input param verbose
 */
template <typename T>
void fit_impl(raft::handle_t &handle, Matrix::RankSizePair **rank_sizes,
              size_t n_parts, Matrix::Data<T> **input, T *components,
              T *singular_vals, paramsTSVD prms, bool verbose) {
  int rank = handle.get_comms().get_rank();

  std::vector<Matrix::RankSizePair *> ranksAndSizes(rank_sizes,
                                                    rank_sizes + n_parts);

  std::vector<Matrix::Data<T> *> input_data(input, input + n_parts);
  Matrix::PartDescriptor input_desc(prms.n_rows, prms.n_cols, ranksAndSizes,
                                    rank);

  // TODO: These streams should come from raft::handle_t
  int n_streams = n_parts;
  cudaStream_t streams[n_streams];
  for (int i = 0; i < n_streams; i++) {
    CUDA_CHECK(cudaStreamCreate(&streams[i]));
  }

  fit_impl(handle, input_data, input_desc, components, singular_vals, prms,
           streams, n_streams, verbose);

  for (int i = 0; i < n_streams; i++) {
    CUDA_CHECK(cudaStreamSynchronize(streams[i]));
  }

  for (int i = 0; i < n_streams; i++) {
    CUDA_CHECK(cudaStreamDestroy(streams[i]));
  }
}

template <typename T>
void transform_impl(raft::handle_t &handle,
                    std::vector<Matrix::Data<T> *> &input,
                    Matrix::PartDescriptor input_desc, T *components,
                    std::vector<Matrix::Data<T> *> &trans_input,
                    paramsTSVD prms, cudaStream_t *streams, int n_streams,
                    bool verbose) {
  int rank = handle.get_comms().get_rank();

  cublasHandle_t cublas_h = handle.get_cublas_handle();
  const auto allocator = handle.get_device_allocator();

  std::vector<Matrix::RankSizePair *> local_blocks =
    input_desc.blocksOwnedBy(rank);

  for (int i = 0; i < input.size(); i++) {
    int si = i % n_streams;

    T alpha = T(1);
    T beta = T(0);
    raft::linalg::gemm(handle, input[i]->ptr, local_blocks[i]->size,
                       size_t(prms.n_cols), components, trans_input[i]->ptr,
                       local_blocks[i]->size, int(prms.n_components),
                       CUBLAS_OP_N, CUBLAS_OP_T, alpha, beta, streams[si]);
  }

  for (int i = 0; i < n_streams; i++) {
    CUDA_CHECK(cudaStreamSynchronize(streams[i]));
  }
}

/**
 * @brief performs MNMG transform operation for the tsvd.
 * @input param handle: the internal cuml handle object
 * @input param rank_sizes: includes all the partition size information for the rank
 * @input param n_parts: number of partitions
 * @input param input: input data
 * @input param components: principal components of the input data
 * @output param trans_input: transformed input data
 * @input param prms: data structure that includes all the parameters from input size to algorithm
 * @input param verbose
 */
template <typename T>
void transform_impl(raft::handle_t &handle, Matrix::RankSizePair **rank_sizes,
                    size_t n_parts, Matrix::Data<T> **input, T *components,
                    Matrix::Data<T> **trans_input, paramsTSVD prms,
                    bool verbose) {
  int rank = handle.get_comms().get_rank();

  std::vector<Matrix::RankSizePair *> ranksAndSizes(rank_sizes,
                                                    rank_sizes + n_parts);
  std::vector<Matrix::Data<T> *> input_data(input, input + n_parts);
  Matrix::PartDescriptor input_desc(prms.n_rows, prms.n_cols, ranksAndSizes,
                                    rank);
  std::vector<Matrix::Data<T> *> trans_data(trans_input, trans_input + n_parts);

  // TODO: These streams should come from raft::handle_t
  int n_streams = n_parts;
  cudaStream_t streams[n_streams];
  for (int i = 0; i < n_streams; i++) {
    CUDA_CHECK(cudaStreamCreate(&streams[i]));
  }

  transform_impl(handle, input_data, input_desc, components, trans_data, prms,
                 streams, n_streams, verbose);

  for (int i = 0; i < n_streams; i++) {
    CUDA_CHECK(cudaStreamSynchronize(streams[i]));
  }

  for (int i = 0; i < n_streams; i++) {
    CUDA_CHECK(cudaStreamDestroy(streams[i]));
  }
}

template <typename T>
void inverse_transform_impl(raft::handle_t &handle,
                            std::vector<Matrix::Data<T> *> &trans_input,
                            Matrix::PartDescriptor trans_input_desc,
                            T *components,
                            std::vector<Matrix::Data<T> *> &input,
                            paramsTSVD prms, cudaStream_t *streams,
                            int n_streams, bool verbose) {
  cublasHandle_t cublas_h = handle.get_cublas_handle();
  const auto allocator = handle.get_device_allocator();
  std::vector<Matrix::RankSizePair *> local_blocks =
    trans_input_desc.partsToRanks;

  for (int i = 0; i < local_blocks.size(); i++) {
    int si = i % n_streams;
    T alpha = T(1);
    T beta = T(0);

    raft::linalg::gemm(handle, trans_input[i]->ptr, local_blocks[i]->size,
                       size_t(prms.n_components), components, input[i]->ptr,
                       local_blocks[i]->size, prms.n_cols, CUBLAS_OP_N,
                       CUBLAS_OP_N, alpha, beta, streams[si]);
  }

  for (int i = 0; i < n_streams; i++) {
    CUDA_CHECK(cudaStreamSynchronize(streams[i]));
  }
}

/**
 * @brief performs MNMG inverse transform operation for the output.
 * @input param handle: the internal cuml handle object
 * @input param rank_sizes: includes all the partition size information for the rank
 * @input param n_parts: number of partitions
 * @input param trans_input: transformed input data
 * @input param components: principal components of the input data
 * @output param input: input data
 * @input param prms: data structure that includes all the parameters from input size to algorithm
 * @input param verbose
 */
template <typename T>
void inverse_transform_impl(raft::handle_t &handle,
                            Matrix::RankSizePair **rank_sizes, size_t n_parts,
                            Matrix::Data<T> **trans_input, T *components,
                            Matrix::Data<T> **input, paramsTSVD prms,
                            bool verbose) {
  int rank = handle.get_comms().get_rank();

  std::vector<Matrix::RankSizePair *> ranksAndSizes(rank_sizes,
                                                    rank_sizes + n_parts);
  Matrix::PartDescriptor trans_desc(prms.n_rows, prms.n_components,
                                    ranksAndSizes, rank);
  std::vector<Matrix::Data<T> *> trans_data(trans_input, trans_input + n_parts);

  std::vector<Matrix::Data<T> *> input_data(input, input + n_parts);

  // TODO: These streams should come from raft::handle_t
  int n_streams = n_parts;
  cudaStream_t streams[n_streams];
  for (int i = 0; i < n_streams; i++) {
    CUDA_CHECK(cudaStreamCreate(&streams[i]));
  }

  inverse_transform_impl(handle, trans_data, trans_desc, components, input_data,
                         prms, streams, n_streams, verbose);

  for (int i = 0; i < n_streams; i++) {
    CUDA_CHECK(cudaStreamSynchronize(streams[i]));
  }

  for (int i = 0; i < n_streams; i++) {
    CUDA_CHECK(cudaStreamDestroy(streams[i]));
  }
}

/**
 * @brief performs MNMG fit and transform operation for the tsvd.
 * @input param handle: the internal cuml handle object
 * @input param rank_sizes: includes all the partition size information for the rank
 * @input param n_parts: number of partitions
 * @input param input: input data
 * @output param trans_input: transformed input data
 * @output param components: principal components of the input data
 * @output param explained_var: explained var
 * @output param explained_var_ratio: the explained var ratio
 * @output param singular_vals: singular values of the data
 * @input param prms: data structure that includes all the parameters from input size to algorithm
 * @input param verbose
 */
template <typename T>
void fit_transform_impl(raft::handle_t &handle,
                        std::vector<Matrix::Data<T> *> &input_data,
                        Matrix::PartDescriptor &input_desc,
                        std::vector<Matrix::Data<T> *> &trans_data,
                        Matrix::PartDescriptor &trans_desc, T *components,
                        T *explained_var, T *explained_var_ratio,
                        T *singular_vals, paramsTSVD prms, bool verbose) {
  int rank = handle.get_comms().get_rank();

  // TODO: These streams should come from raft::handle_t
  int n_streams = input_desc.blocksOwnedBy(rank).size();
  ;
  cudaStream_t streams[n_streams];
  for (int i = 0; i < n_streams; i++) {
    CUDA_CHECK(cudaStreamCreate(&streams[i]));
  }

  fit_impl(handle, input_data, input_desc, components, singular_vals, prms,
           streams, n_streams, verbose);

  transform_impl(handle, input_data, input_desc, components, trans_data, prms,
                 streams, n_streams, verbose);

  PCA::opg::sign_flip(handle, trans_data, input_desc, components,
                      prms.n_components, streams, n_streams);

  device_buffer<T> mu_trans(handle.get_device_allocator(), streams[0],
                            prms.n_components);
  Matrix::Data<T> mu_trans_data{mu_trans.data(), size_t(prms.n_components)};

  Stats::opg::mean(handle, mu_trans_data, trans_data, trans_desc, streams,
                   n_streams);

  Matrix::Data<T> explained_var_data{explained_var, size_t(prms.n_components)};

  Stats::opg::var(handle, explained_var_data, trans_data, trans_desc,
                  mu_trans_data.ptr, streams, n_streams);

  device_buffer<T> mu(handle.get_device_allocator(), streams[0], prms.n_rows);
  Matrix::Data<T> mu_data{mu.data(), size_t(prms.n_rows)};

  Stats::opg::mean(handle, mu_data, input_data, input_desc, streams, n_streams);

  device_buffer<T> var_input(handle.get_device_allocator(), streams[0],
                             prms.n_rows);
  Matrix::Data<T> var_input_data{var_input.data(), size_t(prms.n_rows)};

  Stats::opg::var(handle, var_input_data, input_data, input_desc, mu_data.ptr,
                  streams, n_streams);

  device_buffer<T> total_vars(handle.get_device_allocator(), streams[0], 1);
  raft::stats::sum(total_vars.data(), var_input_data.ptr, 1, prms.n_cols, false,
                   streams[0]);

  T total_vars_h;
  raft::update_host(&total_vars_h, total_vars.data(), 1, streams[0]);
  CUDA_CHECK(cudaStreamSynchronize(streams[0]));
  T scalar = T(1) / total_vars_h;

  raft::linalg::scalarMultiply(explained_var_ratio, explained_var, scalar,
                               prms.n_components, streams[0]);

  for (int i = 0; i < n_streams; i++) {
    CUDA_CHECK(cudaStreamSynchronize(streams[i]));
  }

  for (int i = 0; i < n_streams; i++) {
    CUDA_CHECK(cudaStreamDestroy(streams[i]));
  }
}

void fit(raft::handle_t &handle, Matrix::RankSizePair **rank_sizes,
         size_t n_parts, Matrix::floatData_t **input, float *components,
         float *singular_vals, paramsTSVD prms, bool verbose) {
  fit_impl(handle, rank_sizes, n_parts, input, components, singular_vals, prms,
           verbose);
}

void fit(raft::handle_t &handle, Matrix::RankSizePair **rank_sizes,
         size_t n_parts, Matrix::doubleData_t **input, double *components,
         double *singular_vals, paramsTSVD prms, bool verbose) {
  fit_impl(handle, rank_sizes, n_parts, input, components, singular_vals, prms,
           verbose);
}

void fit_transform(raft::handle_t &handle,
                   std::vector<Matrix::Data<float> *> &input_data,
                   Matrix::PartDescriptor &input_desc,
                   std::vector<Matrix::Data<float> *> &trans_data,
                   Matrix::PartDescriptor &trans_desc, float *components,
                   float *explained_var, float *explained_var_ratio,
                   float *singular_vals, paramsTSVD prms, bool verbose) {
  fit_transform_impl(handle, input_data, input_desc, trans_data, trans_desc,
                     components, explained_var, explained_var_ratio,
                     singular_vals, prms, verbose);
}

void fit_transform(raft::handle_t &handle,
                   std::vector<Matrix::Data<double> *> &input_data,
                   Matrix::PartDescriptor &input_desc,
                   std::vector<Matrix::Data<double> *> &trans_data,
                   Matrix::PartDescriptor &trans_desc, double *components,
                   double *explained_var, double *explained_var_ratio,
                   double *singular_vals, paramsTSVD prms, bool verbose) {
  fit_transform_impl(handle, input_data, input_desc, trans_data, trans_desc,
                     components, explained_var, explained_var_ratio,
                     singular_vals, prms, verbose);
}

void transform(raft::handle_t &handle, Matrix::RankSizePair **rank_sizes,
               size_t n_parts, Matrix::Data<float> **input, float *components,
               Matrix::Data<float> **trans_input, paramsTSVD prms,
               bool verbose) {
  transform_impl(handle, rank_sizes, n_parts, input, components, trans_input,
                 prms, verbose);
}

void transform(raft::handle_t &handle, Matrix::RankSizePair **rank_sizes,
               size_t n_parts, Matrix::Data<double> **input, double *components,
               Matrix::Data<double> **trans_input, paramsTSVD prms,
               bool verbose) {
  transform_impl(handle, rank_sizes, n_parts, input, components, trans_input,
                 prms, verbose);
}

void inverse_transform(raft::handle_t &handle,
                       Matrix::RankSizePair **rank_sizes, size_t n_parts,
                       Matrix::Data<float> **trans_input, float *components,
                       Matrix::Data<float> **input, paramsTSVD prms,
                       bool verbose) {
  inverse_transform_impl(handle, rank_sizes, n_parts, trans_input, components,
                         input, prms, verbose);
}

void inverse_transform(raft::handle_t &handle,
                       Matrix::RankSizePair **rank_sizes, size_t n_parts,
                       Matrix::Data<double> **trans_input, double *components,
                       Matrix::Data<double> **input, paramsTSVD prms,
                       bool verbose) {
  inverse_transform_impl(handle, rank_sizes, n_parts, trans_input, components,
                         input, prms, verbose);
}

}  // namespace opg
}  // namespace TSVD
}  // namespace ML

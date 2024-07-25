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

#include "shuffle.h"

#include <cuml/linear_model/preprocess_mg.hpp>
#include <cuml/solvers/cd_mg.hpp>

#include <cumlprims/opg/linalg/mv_aTb.hpp>
#include <cumlprims/opg/linalg/norm.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>
#include <raft/linalg/add.cuh>
#include <raft/linalg/eltwise.cuh>
#include <raft/linalg/gemm.cuh>
#include <raft/linalg/multiply.cuh>
#include <raft/linalg/subtract.cuh>
#include <raft/matrix/math.cuh>
#include <raft/matrix/matrix.cuh>
#include <raft/util/cuda_utils.cuh>
#include <raft/util/cudart_utils.hpp>

#include <functions/softThres.cuh>

#include <cstddef>

using namespace MLCommon;

namespace ML {
namespace CD {
namespace opg {

template <typename T>
void fit_impl(raft::handle_t& handle,
              std::vector<Matrix::Data<T>*>& input_data,
              Matrix::PartDescriptor& input_desc,
              std::vector<Matrix::Data<T>*>& labels,
              T* coef,
              T* intercept,
              bool fit_intercept,
              bool normalize,
              int epochs,
              T alpha,
              T l1_ratio,
              bool shuffle,
              T tol,
              cudaStream_t* streams,
              int n_streams,
              bool verbose)
{
  const auto& comm = handle.get_comms();

  std::vector<Matrix::RankSizePair*> partsToRanks = input_desc.blocksOwnedBy(comm.get_rank());

  size_t total_M = 0.0;
  for (std::size_t i = 0; i < partsToRanks.size(); i++) {
    total_M += partsToRanks[i]->size;
  }

  rmm::device_uvector<T> pred(total_M, streams[0]);
  rmm::device_uvector<T> residual(total_M, streams[0]);
  rmm::device_uvector<T> squared(input_desc.N, streams[0]);
  rmm::device_uvector<T> mu_input(0, streams[0]);
  rmm::device_uvector<T> norm2_input(0, streams[0]);
  rmm::device_uvector<T> mu_labels(0, streams[0]);

  std::vector<T> h_coef(input_desc.N, T(0));

  if (fit_intercept) {
    mu_input.resize(input_desc.N, streams[0]);
    mu_labels.resize(1, streams[0]);
    if (normalize) { norm2_input.resize(input_desc.N, streams[0]); }

    GLM::opg::preProcessData(handle,
                             input_data,
                             input_desc,
                             labels,
                             mu_input.data(),
                             mu_labels.data(),
                             norm2_input.data(),
                             fit_intercept,
                             normalize,
                             streams,
                             n_streams,
                             verbose);
  }

  std::vector<int> ri(input_desc.N);
  std::mt19937 g(rand());

  size_t memsize = input_desc.N * sizeof(int);
  int* ri_h      = (int*)malloc(memsize);
  RAFT_CUDA_TRY(cudaHostRegister(ri_h, memsize, cudaHostRegisterDefault));

  if (comm.get_rank() == 0) {
    ML::Solver::initShuffle(ri, g);
    for (std::size_t i = 0; i < input_desc.N; i++) {
      ri_h[i] = ri[i];
    }
  }

  comm.bcast(ri_h, input_desc.N, 0, streams[0]);
  comm.sync_stream(streams[0]);

  T l2_alpha = (1 - l1_ratio) * alpha * input_desc.M;
  alpha      = l1_ratio * alpha * input_desc.M;

  if (normalize) {
    T scalar = T(1.0) + l2_alpha;
    raft::matrix::setValue(squared.data(), squared.data(), scalar, input_desc.N, streams[0]);
  } else {
    Matrix::Data<T> squared_data{squared.data(), size_t(input_desc.N)};
    LinAlg::opg::colNorm2NoSeq(handle, squared_data, input_data, input_desc, streams, n_streams);
    raft::linalg::addScalar(squared.data(), squared.data(), l2_alpha, input_desc.N, streams[0]);
  }

  std::vector<Matrix::Data<T>*> input_data_temp;
  Matrix::PartDescriptor input_desc_temp = input_desc;
  input_desc_temp.N                      = size_t(1);
  std::vector<Matrix::Data<T>*> residual_temp;
  Matrix::Data<T> coef_loc_data;

  T* rs = residual.data();
  for (std::size_t i = 0; i < partsToRanks.size(); i++) {
    raft::copy(rs, labels[i]->ptr, partsToRanks[i]->size, streams[0]);

    Matrix::Data<T>* rs_data = new Matrix::Data<T>();
    rs_data->ptr             = rs;
    rs_data->totalSize       = partsToRanks[i]->size;
    residual_temp.push_back(rs_data);

    Matrix::Data<T>* temp_data = new Matrix::Data<T>();
    temp_data->totalSize       = partsToRanks[i]->size;
    input_data_temp.push_back(temp_data);

    rs += partsToRanks[i]->size;
  }

  for (int i = 0; i < epochs; i++) {
    if (i > 0 && shuffle) {
      if (comm.get_rank() == 0) {
        Solver::shuffle(ri, g);
        for (std::size_t k = 0; k < input_desc.N; k++) {
          ri_h[k] = ri[k];
        }
      }

      comm.bcast(ri_h, input_desc.N, 0, streams[0]);
      comm.sync_stream(streams[0]);
    }

    T coef_max   = 0.0;
    T d_coef_max = 0.0;
    T coef_prev  = 0.0;

    for (std::size_t j = 0; j < input_desc.N; j++) {
      int ci         = ri_h[j];
      T* coef_loc    = coef + ci;
      T* squared_loc = squared.data() + ci;
      T* input_col_loc;
      T* pred_loc     = pred.data();
      T* residual_loc = residual.data();

      for (std::size_t k = 0; k < input_data.size(); k++) {
        input_col_loc = input_data[k]->ptr + (ci * partsToRanks[k]->size);

        input_data_temp[k]->ptr       = input_col_loc;
        input_data_temp[k]->totalSize = partsToRanks[k]->size;

        raft::linalg::multiplyScalar(
          pred_loc, input_col_loc, h_coef[ci], partsToRanks[k]->size, streams[k % n_streams]);

        raft::linalg::add(
          residual_loc, residual_loc, pred_loc, partsToRanks[k]->size, streams[k % n_streams]);

        pred_loc     = pred_loc + partsToRanks[k]->size;
        residual_loc = residual_loc + partsToRanks[k]->size;
      }

      for (int k = 0; k < n_streams; k++) {
        handle.sync_stream(streams[k]);
      }

      coef_loc_data.ptr       = coef_loc;
      coef_loc_data.totalSize = size_t(1);
      LinAlg::opg::mv_aTb(
        handle, coef_loc_data, input_data_temp, input_desc_temp, residual_temp, streams, n_streams);

      if (l1_ratio > T(0.0)) Functions::softThres(coef_loc, coef_loc, alpha, 1, streams[0]);

      raft::linalg::eltwiseDivideCheckZero(coef_loc, coef_loc, squared_loc, 1, streams[0]);

      coef_prev = h_coef[ci];
      raft::update_host(&(h_coef[ci]), coef_loc, 1, streams[0]);
      handle.sync_stream(streams[0]);

      T diff = abs(coef_prev - h_coef[ci]);

      if (diff > d_coef_max) d_coef_max = diff;

      if (abs(h_coef[ci]) > coef_max) coef_max = abs(h_coef[ci]);

      pred_loc     = pred.data();
      residual_loc = residual.data();

      for (std::size_t k = 0; k < input_data.size(); k++) {
        input_col_loc = input_data[k]->ptr + (ci * partsToRanks[k]->size);

        raft::linalg::multiplyScalar(
          pred_loc, input_col_loc, h_coef[ci], partsToRanks[k]->size, streams[k % n_streams]);

        raft::linalg::subtract(
          residual_loc, residual_loc, pred_loc, partsToRanks[k]->size, streams[k % n_streams]);

        pred_loc     = pred_loc + partsToRanks[k]->size;
        residual_loc = residual_loc + partsToRanks[k]->size;
      }

      for (int k = 0; k < n_streams; k++) {
        handle.sync_stream(streams[k]);
      }
    }

    bool flag_continue = true;
    if (coef_max == T(0)) { flag_continue = false; }

    if ((d_coef_max / coef_max) < tol) { flag_continue = false; }

    if (!flag_continue) { break; }
  }

  RAFT_CUDA_TRY(cudaHostUnregister(ri_h));
  free(ri_h);

  for (std::size_t i = 0; i < partsToRanks.size(); i++) {
    delete residual_temp[i];
    delete input_data_temp[i];
  }

  if (fit_intercept) {
    GLM::opg::postProcessData(handle,
                              input_data,
                              input_desc,
                              labels,
                              coef,
                              intercept,
                              mu_input.data(),
                              mu_labels.data(),
                              norm2_input.data(),
                              fit_intercept,
                              normalize,
                              streams,
                              n_streams,
                              verbose);
  } else {
    *intercept = T(0);
  }
}

/**
 * @brief performs MNMG fit operation for the ols
 * @input param handle: the internal cuml handle object
 * @input param rank_sizes: includes all the partition size information for the rank
 * @input param n_parts: number of partitions
 * @input param input: input data
 * @input param labels: labels data
 * @output param coef: learned regression coefficients
 * @output param intercept: intercept value
 * @input param fit_intercept: fit intercept or not
 * @input param normalize: normalize the data or not
 * @input param verbose
 */
template <typename T>
void fit_impl(raft::handle_t& handle,
              std::vector<Matrix::Data<T>*>& input_data,
              Matrix::PartDescriptor& input_desc,
              std::vector<Matrix::Data<T>*>& labels,
              T* coef,
              T* intercept,
              bool fit_intercept,
              bool normalize,
              int epochs,
              T alpha,
              T l1_ratio,
              bool shuffle,
              T tol,
              bool verbose)
{
  int rank = handle.get_comms().get_rank();

  // TODO: These streams should come from raft::handle_t
  // Tracking issue: https://github.com/rapidsai/cuml/issues/2470

  int n_streams = input_desc.blocksOwnedBy(rank).size();
  ;
  cudaStream_t streams[n_streams];
  for (int i = 0; i < n_streams; i++) {
    RAFT_CUDA_TRY(cudaStreamCreate(&streams[i]));
  }

  fit_impl(handle,
           input_data,
           input_desc,
           labels,
           coef,
           intercept,
           fit_intercept,
           normalize,
           epochs,
           alpha,
           l1_ratio,
           shuffle,
           tol,
           streams,
           n_streams,
           verbose);

  for (int i = 0; i < n_streams; i++) {
    handle.sync_stream(streams[i]);
  }

  for (int i = 0; i < n_streams; i++) {
    RAFT_CUDA_TRY(cudaStreamDestroy(streams[i]));
  }
}

template <typename T>
void predict_impl(raft::handle_t& handle,
                  std::vector<Matrix::Data<T>*>& input_data,
                  Matrix::PartDescriptor& input_desc,
                  T* coef,
                  T intercept,
                  std::vector<Matrix::Data<T>*>& preds,
                  cudaStream_t* streams,
                  int n_streams,
                  bool verbose)
{
  std::vector<Matrix::RankSizePair*> local_blocks = input_desc.partsToRanks;
  T alpha                                         = T(1);
  T beta                                          = T(0);

  for (std::size_t i = 0; i < input_data.size(); i++) {
    int si = i % n_streams;
    raft::linalg::gemm(handle,
                       input_data[i]->ptr,
                       local_blocks[i]->size,
                       input_desc.N,
                       coef,
                       preds[i]->ptr,
                       local_blocks[i]->size,
                       size_t(1),
                       CUBLAS_OP_N,
                       CUBLAS_OP_N,
                       alpha,
                       beta,
                       streams[si]);

    raft::linalg::addScalar(
      preds[i]->ptr, preds[i]->ptr, intercept, local_blocks[i]->size, streams[si]);
  }
}

template <typename T>
void predict_impl(raft::handle_t& handle,
                  Matrix::RankSizePair** rank_sizes,
                  size_t n_parts,
                  Matrix::Data<T>** input,
                  size_t n_rows,
                  size_t n_cols,
                  T* coef,
                  T intercept,
                  Matrix::Data<T>** preds,
                  bool verbose)
{
  int rank = handle.get_comms().get_rank();

  std::vector<Matrix::RankSizePair*> ranksAndSizes(rank_sizes, rank_sizes + n_parts);
  std::vector<Matrix::Data<T>*> input_data(input, input + n_parts);
  Matrix::PartDescriptor input_desc(n_rows, n_cols, ranksAndSizes, rank);
  std::vector<Matrix::Data<T>*> preds_data(preds, preds + n_parts);

  // TODO: These streams should come from raft::handle_t
  // Tracking issue: https://github.com/rapidsai/cuml/issues/2470
  int n_streams = n_parts;
  cudaStream_t streams[n_streams];
  for (int i = 0; i < n_streams; i++) {
    RAFT_CUDA_TRY(cudaStreamCreate(&streams[i]));
  }

  predict_impl(
    handle, input_data, input_desc, coef, intercept, preds_data, streams, n_streams, verbose);

  for (int i = 0; i < n_streams; i++) {
    handle.sync_stream(streams[i]);
  }

  for (int i = 0; i < n_streams; i++) {
    RAFT_CUDA_TRY(cudaStreamDestroy(streams[i]));
  }
}

void fit(raft::handle_t& handle,
         std::vector<Matrix::Data<float>*>& input_data,
         Matrix::PartDescriptor& input_desc,
         std::vector<Matrix::Data<float>*>& labels,
         float* coef,
         float* intercept,
         bool fit_intercept,
         bool normalize,
         int epochs,
         float alpha,
         float l1_ratio,
         bool shuffle,
         float tol,
         bool verbose)
{
  fit_impl(handle,
           input_data,
           input_desc,
           labels,
           coef,
           intercept,
           fit_intercept,
           normalize,
           epochs,
           alpha,
           l1_ratio,
           shuffle,
           tol,
           verbose);
}

void fit(raft::handle_t& handle,
         std::vector<Matrix::Data<double>*>& input_data,
         Matrix::PartDescriptor& input_desc,
         std::vector<Matrix::Data<double>*>& labels,
         double* coef,
         double* intercept,
         bool fit_intercept,
         bool normalize,
         int epochs,
         double alpha,
         double l1_ratio,
         bool shuffle,
         double tol,
         bool verbose)
{
  fit_impl(handle,
           input_data,
           input_desc,
           labels,
           coef,
           intercept,
           fit_intercept,
           normalize,
           epochs,
           alpha,
           l1_ratio,
           shuffle,
           tol,
           verbose);
}

void predict(raft::handle_t& handle,
             Matrix::RankSizePair** rank_sizes,
             size_t n_parts,
             Matrix::Data<float>** input,
             size_t n_rows,
             size_t n_cols,
             float* coef,
             float intercept,
             Matrix::Data<float>** preds,
             bool verbose)
{
  predict_impl(handle, rank_sizes, n_parts, input, n_rows, n_cols, coef, intercept, preds, verbose);
}

void predict(raft::handle_t& handle,
             Matrix::RankSizePair** rank_sizes,
             size_t n_parts,
             Matrix::Data<double>** input,
             size_t n_rows,
             size_t n_cols,
             double* coef,
             double intercept,
             Matrix::Data<double>** preds,
             bool verbose)
{
  predict_impl(handle, rank_sizes, n_parts, input, n_rows, n_cols, coef, intercept, preds, verbose);
}

}  // namespace opg
}  // namespace CD
}  // namespace ML

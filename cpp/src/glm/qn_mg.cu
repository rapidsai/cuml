/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include "qn/mg/qn_mg.cuh"
#include "qn/simple_mat/dense.hpp"
#include <cuda_runtime.h>
#include <cuml/common/logger.hpp>
#include <cuml/linear_model/qn.h>
#include <cuml/linear_model/qn_mg.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/error.hpp>
#include <raft/core/handle.hpp>
#include <raft/label/classlabels.cuh>
#include <raft/linalg/divide.cuh>
#include <raft/linalg/matrix_vector.cuh>
#include <raft/linalg/matrix_vector_op.cuh>
#include <raft/linalg/sqrt.cuh>
#include <raft/matrix/math.hpp>
#include <raft/stats/mean_center.cuh>
#include <raft/stats/stddev.cuh>
#include <raft/stats/sum.cuh>
#include <raft/util/cudart_utils.hpp>
#include <vector>
using namespace MLCommon;

#include <cumlprims/opg/matrix/math.hpp>
#include <cumlprims/opg/stats/mean.hpp>
#include <cumlprims/opg/stats/mean_center.hpp>
#include <cumlprims/opg/stats/stddev.hpp>

namespace ML {
namespace GLM {
namespace opg {

template <typename T>
std::vector<T> distinct_mg(const raft::handle_t& handle, T* y, size_t n)
{
  cudaStream_t stream              = handle.get_stream();
  raft::comms::comms_t const& comm = raft::resource::get_comms(handle);
  int rank                         = comm.get_rank();
  int n_ranks                      = comm.get_size();

  rmm::device_uvector<T> unique_y(0, stream);
  raft::label::getUniquelabels(unique_y, y, n, stream);

  rmm::device_uvector<size_t> recv_counts(n_ranks, stream);
  auto send_count = raft::make_device_scalar<size_t>(handle, unique_y.size());
  comm.allgather(send_count.data_handle(), recv_counts.data(), 1, stream);
  comm.sync_stream(stream);

  std::vector<size_t> recv_counts_host(n_ranks);
  raft::copy(recv_counts_host.data(), recv_counts.data(), n_ranks, stream);

  std::vector<size_t> displs(n_ranks);
  size_t pos = 0;
  for (int i = 0; i < n_ranks; ++i) {
    displs[i] = pos;
    pos += recv_counts_host[i];
  }

  rmm::device_uvector<T> recv_buff(displs.back() + recv_counts_host.back(), stream);
  comm.allgatherv(
    unique_y.data(), recv_buff.data(), recv_counts_host.data(), displs.data(), stream);
  comm.sync_stream(stream);

  rmm::device_uvector<T> global_unique_y(0, stream);
  int n_distinct =
    raft::label::getUniquelabels(global_unique_y, recv_buff.data(), recv_buff.size(), stream);

  std::vector<T> global_unique_y_host(global_unique_y.size());
  raft::copy(global_unique_y_host.data(), global_unique_y.data(), global_unique_y.size(), stream);

  return global_unique_y_host;
}

template <typename T>
void standardize_impl(const raft::handle_t& handle,
                      T* input_data,
                      const Matrix::PartDescriptor& input_desc,
                      bool col_major,
                      T* mean_vector,
                      T* stddev_vector)
{
  int D        = input_desc.N;
  int rank     = input_desc.rank;
  int num_rows = input_desc.totalElementsOwnedBy(rank);
  auto stream  = handle.get_stream();
  auto& comm   = handle.get_comms();

  raft::stats::sum(mean_vector, input_data, D, num_rows, !col_major, stream);
  T weight = T(1) / T(input_desc.M);
  raft::linalg::multiplyScalar(mean_vector, mean_vector, weight, D, stream);
  comm.allreduce(mean_vector, mean_vector, D, raft::comms::op_t::SUM, stream);
  comm.sync_stream(stream);
  // auto log_mean_str = raft::arr2Str(mean_vector, D, "mean_str", handle.get_stream(), 8);
  // CUML_LOG_DEBUG("rank %d mean vector %s", rank, log_mean_str.c_str());

  std::vector<T> cpu_mean(D);
  raft::copy(cpu_mean.data(), mean_vector, D, stream);
  CUML_LOG_DEBUG("rank %d cpu_mean vector %0.8f and %0.8f", rank, cpu_mean[0], cpu_mean[1]);

  raft::stats::vars(stddev_vector, input_data, mean_vector, D, num_rows, false, !col_major, stream);
  weight = T(1) * num_rows / T(input_desc.M);
  raft::linalg::multiplyScalar(stddev_vector, stddev_vector, weight, D, stream);
  comm.allreduce(stddev_vector, stddev_vector, D, raft::comms::op_t::SUM, stream);
  comm.sync_stream(stream);
  // auto log_var_str = raft::arr2Str(stddev_vector, D, "var_str", handle.get_stream(), 8);
  // CUML_LOG_DEBUG("rank %d var vector %s", rank, log_var_str.c_str());
  std::vector<T> cpu_var(D);
  raft::copy(cpu_var.data(), stddev_vector, D, stream);
  CUML_LOG_DEBUG("rank %d cpu_var vector %0.8f and %0.8f", rank, cpu_var[0], cpu_var[1]);

  raft::linalg::sqrt(stddev_vector, stddev_vector, D, handle.get_stream());

  // auto log_std_str = raft::arr2Str(stddev_vector, D, "std_str", handle.get_stream(), 8);
  // CUML_LOG_DEBUG("rank %d std vector %s", rank, log_std_str.c_str());
  std::vector<T> cpu_std(D);
  raft::copy(cpu_std.data(), stddev_vector, D, stream);
  CUML_LOG_DEBUG("rank %d cpu_std vector %0.8f and %0.8f", rank, cpu_std[0], cpu_std[1]);

  raft::stats::meanCenter(
    input_data, input_data, mean_vector, D, num_rows, !col_major, !col_major, stream);
  // auto log_stddata_str = raft::arr2Str(input_data, num_rows * D, "stddata", stream);
  // CUML_LOG_DEBUG("rank %d stddev vector %s", rank, log_stddata_str.c_str());

  raft::matrix::matrixVectorBinaryDivSkipZero(
    input_data, stddev_vector, num_rows, D, !col_major, !col_major, stream);
  // log_stddata_str = raft::arr2Str(input_data, num_rows * D, "stddata", handle.get_stream());
  // CUML_LOG_DEBUG("rank %d stddev vector %s", rank, log_stddata_str.c_str());
}

template <typename T>
void undo_standardize_impl(const raft::handle_t& handle,
                           std::vector<Matrix::Data<T>*>& input_data,
                           const Matrix::PartDescriptor& input_desc,
                           bool col_major,
                           T* mean_vector,
                           T* stddev_vector)
{
  auto n_streams = input_desc.blocksOwnedBy(input_desc.rank).size();
  cudaStream_t streams[n_streams];
  for (std::uint32_t i = 0; i < n_streams; i++) {
    RAFT_CUDA_TRY(cudaStreamCreate(&streams[i]));
  }

  int D               = input_desc.N;
  bool row_major      = col_major ? false : true;
  bool bcastAlongRows = true;
  Matrix::Data<T> stddev_data(stddev_vector, D);
  Matrix::opg::matrixVectorBinaryMult(input_data,
                                      input_desc,
                                      stddev_data,
                                      row_major,
                                      bcastAlongRows,
                                      handle.get_comms(),
                                      streams,
                                      n_streams);

  Matrix::Data<T> mu_data(mean_vector, D);
  Stats::opg::mean_add(input_data, input_desc, mu_data, handle.get_comms(), streams, n_streams);
}

template <typename T>
void qnFit_impl(const raft::handle_t& handle,
                const qn_params& pams,
                T* X,
                bool X_col_major,
                T* y,
                size_t N,
                size_t D,
                size_t C,
                T* w0,
                T* f,
                int* num_iters,
                size_t n_samples,
                int rank,
                int n_ranks)
{
  auto X_simple = SimpleDenseMat<T>(X, N, D, X_col_major ? COL_MAJOR : ROW_MAJOR);

  ML::GLM::opg::qn_fit_x_mg(handle,
                            pams,
                            X_simple,
                            y,
                            C,
                            w0,
                            f,
                            num_iters,
                            n_samples,
                            rank,
                            n_ranks);  // ignore sample_weight, svr_eps
  return;
}

template <typename T>
void qnFit_impl(raft::handle_t& handle,
                std::vector<Matrix::Data<T>*>& input_data,
                Matrix::PartDescriptor& input_desc,
                std::vector<Matrix::Data<T>*>& labels,
                T* coef,
                const qn_params& pams,
                bool X_col_major,
                bool standardization,
                int n_classes,
                T* f,
                int* num_iters)
{
  RAFT_EXPECTS(input_data.size() == 1,
               "qn_mg.cu currently does not accept more than one input matrix");
  RAFT_EXPECTS(labels.size() == input_data.size(), "labels size does not equal to input_data size");

  auto data_X = input_data[0];
  auto data_y = labels[0];

  ML::Logger::get().setLevel(pams.verbose);
  CUML_LOG_DEBUG(
    "rank %d gets standardization %s", input_desc.rank, standardization ? "true" : "false");
  // std::cout << "rank " << input_desc.rank << " gets standardization " << standardization <<
  // std::endl; int num_elements = input_desc.totalElementsOwnedBy(input_desc.rank); auto data_str =
  // raft::arr2Str(input_data[0]->ptr, num_elements, "data_str", handle.get_stream()); std::cout <<
  // "rank " << input_desc.rank << " gets c_str " << data_str << std::endl; CUML_LOG_DEBUG("rank %d
  // gets #elements %d, and data %s", input_desc.rank, num_elements, data_str.c_str());

  size_t n_samples = 0;
  for (auto p : input_desc.partsToRanks) {
    n_samples += p->size;
  }

  auto stream = handle.get_stream();
  rmm::device_uvector<T> mean_vec(input_desc.N, stream);
  rmm::device_uvector<T> stddev_vec(input_desc.N, stream);
  if (standardization) {
    standardize_impl<T>(
      handle, data_X->ptr, input_desc, X_col_major, mean_vec.data(), stddev_vec.data());
  }

  qnFit_impl<T>(handle,
                pams,
                data_X->ptr,
                X_col_major,
                data_y->ptr,
                input_desc.totalElementsOwnedBy(input_desc.rank),
                input_desc.N,
                n_classes,
                coef,
                f,
                num_iters,
                input_desc.M,
                input_desc.rank,
                input_desc.uniqueRanks().size());

  auto log_coef_str = raft::arr2Str(coef, input_desc.N + pams.fit_intercept, "coef", stream, 16);
  CUML_LOG_DEBUG("rank %d gets coefs %s", input_desc.rank, log_coef_str.c_str());
  if (standardization) {
    // adapt intercepts and coefficients to avoid actual standardization in model inference

    int n_targets =
      ML::GLM::detail::qn_is_classification(pams.loss) && n_classes == 2 ? 1 : n_classes;
    int D = input_desc.N;
    SimpleDenseMat<T> W(coef, n_targets, D + pams.fit_intercept);

    SimpleDenseMat<T> Wweights;
    col_slice(W, Wweights, 0, D);

    auto div_stddev = [] __device__(const T a, const T b) {
      if (b == 0.0) return (T)0.0;
      return a / b;
    };
    raft::linalg::matrixVectorOp(Wweights.data,
                                 Wweights.data,
                                 stddev_vec.data(),
                                 Wweights.n,
                                 Wweights.m,
                                 false,
                                 true,
                                 div_stddev,
                                 stream);

    if (pams.fit_intercept) {
      SimpleVec<T> Wbias;
      col_ref(W, Wbias, D);

      SimpleVec<T> meanVec(mean_vec.data(), mean_vec.size());
      Wbias.assign_gemv(handle, -1, Wweights, false, meanVec, 1, stream);
    }

    auto log_adaptcoef_str =
      raft::arr2Str(coef, input_desc.N + pams.fit_intercept, "adaptcoef", stream, 16);
    CUML_LOG_DEBUG("rank %d gets adapted coefs %s", input_desc.rank, log_adaptcoef_str.c_str());
    // TODO: undo standardization
    undo_standardize_impl<T>(
      handle, input_data, input_desc, X_col_major, mean_vec.data(), stddev_vec.data());

    // int num_elements = input_desc.totalElementsOwnedBy(input_desc.rank);
    // auto log_origindata_str =
    //   raft::arr2Str(input_data[0]->ptr, num_elements * D, "origin data", stream);
    // CUML_LOG_DEBUG("rank %d gets returned dataset %s", input_desc.rank,
    // log_origindata_str.c_str());
  }
}

std::vector<float> getUniquelabelsMG(const raft::handle_t& handle,
                                     Matrix::PartDescriptor& input_desc,
                                     std::vector<Matrix::Data<float>*>& labels)
{
  RAFT_EXPECTS(labels.size() == 1,
               "getUniqueLabelsMG currently does not accept more than one data chunk");
  Matrix::Data<float>* data_y = labels[0];
  int n_rows                  = input_desc.totalElementsOwnedBy(input_desc.rank);
  return distinct_mg<float>(handle, data_y->ptr, n_rows);
}

void qnFit(raft::handle_t& handle,
           std::vector<Matrix::Data<float>*>& input_data,
           Matrix::PartDescriptor& input_desc,
           std::vector<Matrix::Data<float>*>& labels,
           float* coef,
           const qn_params& pams,
           bool X_col_major,
           bool standardization,
           int n_classes,
           float* f,
           int* num_iters)
{
  qnFit_impl<float>(handle,
                    input_data,
                    input_desc,
                    labels,
                    coef,
                    pams,
                    X_col_major,
                    standardization,
                    n_classes,
                    f,
                    num_iters);
}

template <typename T, typename I>
void qnFitSparse_impl(const raft::handle_t& handle,
                      const qn_params& pams,
                      T* X_values,
                      I* X_cols,
                      I* X_row_ids,
                      I X_nnz,
                      T* y,
                      size_t N,
                      size_t D,
                      size_t C,
                      T* w0,
                      T* f,
                      int* num_iters,
                      size_t n_samples,
                      int rank,
                      int n_ranks)
{
  auto X_simple = SimpleSparseMat<T>(X_values, X_cols, X_row_ids, X_nnz, N, D);

  ML::GLM::opg::qn_fit_x_mg(handle,
                            pams,
                            X_simple,
                            y,
                            C,
                            w0,
                            f,
                            num_iters,
                            n_samples,
                            rank,
                            n_ranks);  // ignore sample_weight, svr_eps
  return;
}

void qnFitSparse(raft::handle_t& handle,
                 std::vector<Matrix::Data<float>*>& input_values,
                 int* input_cols,
                 int* input_row_ids,
                 int X_nnz,
                 Matrix::PartDescriptor& input_desc,
                 std::vector<Matrix::Data<float>*>& labels,
                 float* coef,
                 const qn_params& pams,
                 int n_classes,
                 float* f,
                 int* num_iters)
{
  RAFT_EXPECTS(input_values.size() == 1,
               "qn_mg.cu currently does not accept more than one input matrix");

  auto data_input_values = input_values[0];
  auto data_y            = labels[0];

  qnFitSparse_impl<float, int>(handle,
                               pams,
                               data_input_values->ptr,
                               input_cols,
                               input_row_ids,
                               X_nnz,
                               data_y->ptr,
                               input_desc.totalElementsOwnedBy(input_desc.rank),
                               input_desc.N,
                               n_classes,
                               coef,
                               f,
                               num_iters,
                               input_desc.M,
                               input_desc.rank,
                               input_desc.uniqueRanks().size());
}

};  // namespace opg
};  // namespace GLM
};  // namespace ML

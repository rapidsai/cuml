/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include "qn/mg/standardization.cuh"
#include "qn/simple_mat/dense.hpp"

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

#include <cuda_runtime.h>

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
  raft::resource::sync_stream(handle);

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
  raft::resource::sync_stream(handle);

  return global_unique_y_host;
}

template <typename T>
void qnFit_impl(const raft::handle_t& handle,
                const qn_params& pams,
                T* X,
                bool X_col_major,
                bool standardization,
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

  rmm::device_uvector<T> mean_std_buff(4 * D, handle.get_stream());
  Standardizer<T>* stder = NULL;
  if (standardization) stder = new Standardizer(handle, X_simple, n_samples, mean_std_buff);

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
                            n_ranks,
                            stder);  // ignore sample_weight, svr_eps

  if (standardization) {
    int n_targets = ML::GLM::detail::qn_is_classification(pams.loss) && C == 2 ? 1 : C;
    stder->adapt_model_for_linearFwd(handle, w0, n_targets, D, pams.fit_intercept);
    delete stder;
  }

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

  size_t n_samples = 0;
  for (auto p : input_desc.partsToRanks) {
    n_samples += p->size;
  }

  auto stream = handle.get_stream();

  qnFit_impl<T>(handle,
                pams,
                data_X->ptr,
                X_col_major,
                standardization,
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
                      bool standardization,
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
  RAFT_EXPECTS(standardization == false, "standardization for sparse vectors is not supported yet");

  auto X_simple = SimpleSparseMat<T>(X_values, X_cols, X_row_ids, X_nnz, N, D);

  rmm::device_uvector<T> mean_std_buff(4 * D, handle.get_stream());
  Standardizer<T>* stder = NULL;

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
                            n_ranks,
                            stder);  // ignore sample_weight, svr_eps

  if (standardization) {
    int n_targets = ML::GLM::detail::qn_is_classification(pams.loss) && C == 2 ? 1 : C;
    stder->adapt_model_for_linearFwd(handle, w0, n_targets, D, pams.fit_intercept);
    delete stder;
  }

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
                 bool standardization,
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
                               standardization,
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

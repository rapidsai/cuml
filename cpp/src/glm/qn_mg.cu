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
#include <raft/core/error.hpp>
#include <raft/core/handle.hpp>
#include <raft/label/classlabels.cuh>
#include <raft/util/cudart_utils.hpp>
using namespace MLCommon;

#include <iostream>

namespace ML {
namespace GLM {
namespace opg {

template <typename T>
int distinct(const raft::handle_t& handle, T* y, size_t n_rows)
{
  cudaStream_t stream = handle.get_stream();

  std::cout << "distinct: n_rows " << n_rows << std::endl;
  std::cout << raft::arr2Str(y, n_rows, "distinct: y is ", stream) << std::endl;

  rmm::device_uvector<T> unique_labels(0, stream);
  raft::label::getUniquelabels(unique_labels, y, n_rows, stream);
  std::cout << raft::arr2Str(
                 unique_labels.data(), unique_labels.size(), "distinct: unique_labels is ", stream)
            << std::endl;

  raft::comms::comms_t const& communicator = raft::resource::get_comms(handle);
  communicator.allgather(unique_labels.data(), unique_labels.data(), unique_labels.size(), stream);
  communicator.sync_stream(stream);
  std::cout << raft::arr2Str(unique_labels.data(),
                             unique_labels.size(),
                             "distinct: gathered unique_labels is ",
                             stream)
            << std::endl;

  rmm::device_uvector<T> global_unique_labels(0, stream);
  int num_distinct = raft::label::getUniquelabels(
    global_unique_labels, unique_labels.data(), unique_labels.size(), stream);
  std::cout << raft::arr2Str(global_unique_labels.data(),
                             global_unique_labels.size(),
                             "distinct: global_unique_labels is ",
                             stream)
            << std::endl;

  std::cout << "debug num_distinct is: " << num_distinct << std::endl;
  return num_distinct;
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
  /*
  switch (pams.loss) {
    case QN_LOSS_LOGISTIC: {
      RAFT_EXPECTS(
        C == 2,
        "qn_mg.cu: only the LOGISTIC loss is supported currently. The number of classes must be 2");
    } break;
    default: {
      RAFT_EXPECTS(false, "qn_mg.cu: unknown loss function type (id = %d).", pams.loss);
    }
  }
  */

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
}

int qnNumClasses(const raft::handle_t& handle, std::vector<Matrix::Data<float>*>& labels)
{
  RAFT_EXPECTS(labels.size() == 1, "distinct currently does not accept more than one data chunk");

  Matrix::Data<float>* data_y = labels[0];
  return distinct<float>(handle, data_y->ptr, data_y->numElements());
}

void qnFit(raft::handle_t& handle,
           std::vector<Matrix::Data<float>*>& input_data,
           Matrix::PartDescriptor& input_desc,
           std::vector<Matrix::Data<float>*>& labels,
           float* coef,
           const qn_params& pams,
           bool X_col_major,
           int n_classes,
           float* f,
           int* num_iters)
{
  qnFit_impl<float>(
    handle, input_data, input_desc, labels, coef, pams, X_col_major, n_classes, f, num_iters);
}

};  // namespace opg
};  // namespace GLM
};  // namespace ML

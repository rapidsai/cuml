/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <raft/core/comms.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/core/handle.hpp>
#include <cuml/common/logger.hpp>
#include <cuml/linear_model/qn.h> 
#include "qn/simple_mat/dense.hpp"
#include "qn/qn_util.cuh"
#include "qn/glm_logistic.cuh"
#include "qn/glm_regularizer.cuh"
#include <cuml/linear_model/qn_mg.hpp>
using namespace MLCommon;

#include "qn/glm_base_mg.cuh"

#include <cuda_runtime.h>
#include <iostream>


namespace ML {
namespace GLM {
namespace opg {

/*
void toy(const raft::handle_t &handle,
         std::vector<Matrix::Data<float>*>& input_data,
         Matrix::PartDescriptor& input_desc,
         std::vector<Matrix::Data<float>*>& labels,
         float* coef) 
{
  ASSERT(input_data.size() == 1, "qn_mg.cu currently does not accept more than one input matrix");
  ASSERT(labels.size() == input_data.size(), "labels size does not input_data size ");

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  std::cout << "entered toy:input_data.size() " << input_data.size() << std::endl;
  auto X = input_data[0];
  std::cout << "X.numElements: " << X->numElements() << ", totalSize: " << X->totalSize << std::endl;
  std::cout << "X: " << raft::arr2Str(X->ptr, 4, "first X", stream).c_str() << std::endl;

  auto y = labels[0];
  std::cout << "y.numElements: " << y->numElements() << ", totalSize: " << y->totalSize << std::endl;

  std::cout << "first_y.ptr: " << raft::arr2Str(y->ptr, 2, "first_y ", stream).c_str() << std::endl;

  int N = input_desc.M;
  int D = input_desc.N;
  int rank = input_desc.rank;
  int n_ranks = input_desc.partsToRanks.size();
  size_t n_samples = 0;
  for (auto p : input_desc.partsToRanks) {
    n_samples += p->size;

  }

  std::cout << "report: N: " << N << ", D: " << D << ", rank: " << rank << ", n_ranks " << n_ranks << ", n_samples " << n_samples << std::endl;




}
*/

template<typename T>
void qnFit_impl(const raft::handle_t &handle, 
                const qn_params& pams,
                T* X,
                bool X_col_major,
                T *y,
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
  switch (pams.loss) {
    case QN_LOSS_LOGISTIC: {
      ASSERT(C == 2, "qn_mg.cu: logistic loss invalid C");
    } break;
    default: {
      ASSERT(false, "qn_mg.cu: unknown loss function type (id = %d).", pams.loss);
    }
  }

  cudaStream_t stream = raft::resource::get_cuda_stream(handle);
  auto X_simple = SimpleDenseMat<T>(X, N, D, X_col_major? COL_MAJOR : ROW_MAJOR);
  auto y_simple = SimpleVec<T>(y, N);
  SimpleVec<T> coef_simple(w0, D + pams.fit_intercept);

  //std::cout << "rank " << rank << ", N " << N << ", D " << D << ", n_samples " << n_samples << std::endl;
  //std::cout << "ranl " << rank << raft::arr2Str(X_simple.data, N * D, "X_simple", stream) << std::endl;
  //std::cout << "ranl " << rank << raft::arr2Str(y_simple.data, N, "y_simple", stream) << std::endl;
  //std::cout << "ranl " << rank << raft::arr2Str(coef_simple.data, D + pams.fit_intercept, "y_simple", stream) << std::endl;

  ML::GLM::detail::LBFGSParam<T> opt_param(pams);

  // prepare regularizer regularizer_obj
  ML::GLM::detail::LogisticLoss<T> loss_func(handle, D, pams.fit_intercept);
  T l2 = pams.penalty_l2;
  if (pams.penalty_normalized) {
      l2 /= n_samples; // l2 /= 1/X.m
  }
  ML::GLM::detail::Tikhonov<T> reg(l2);
  ML::GLM::detail::RegularizedGLM<T, ML::GLM::detail::LogisticLoss<T>, decltype(reg)> regularizer_obj(&loss_func, &reg);


  // prepare GLMWithDataMG
  int n_targets = C == 2 ? 1 : C;
  rmm::device_uvector<T> tmp(n_targets * N, stream);
  SimpleDenseMat<T> Z(tmp.data(), n_targets, N);
  auto obj_function = GLMWithDataMG(handle, rank, n_ranks, n_samples, &regularizer_obj, X_simple, y_simple, Z);

  // prepare temporary variables fx, k, workspace 
  float fx = -1; 
  int k = -1;
  rmm::device_uvector<float> tmp_workspace(lbfgs_workspace_size(opt_param, coef_simple.len), stream);
  SimpleVec<float> workspace(tmp_workspace.data(), tmp_workspace.size());

  // call min_lbfgs
  min_lbfgs(opt_param, obj_function, coef_simple, fx, &k, workspace, stream, 5);

}

template <typename T>
void qnFit_impl(raft::handle_t& handle,
              std::vector<Matrix::Data<T>*>& input_data,
              Matrix::PartDescriptor& input_desc,
              std::vector<Matrix::Data<T>*>& labels,
              T* coef,
              const qn_params& pams,
              bool X_col_major,
              T* f,
              int* num_iters) 
{
  ASSERT(input_data.size() == 1, "qn_mg.cu currently does not accept more than one input matrix");
  ASSERT(labels.size() == input_data.size(), "labels size does not equal to input_data size");

  auto data_X = input_data[0];
  auto data_y = labels[0];

  size_t n_samples = 0;
  for (auto p : input_desc.partsToRanks) {
    n_samples += p->size;
  }

  qnFit_impl<T>(
    handle,
    pams,
    data_X->ptr,
    X_col_major,
    data_y->ptr,
    input_desc.totalElementsOwnedBy(input_desc.rank),
    input_desc.N, 
    2,  // TODO: support multiple classes
    coef,
    f,
    num_iters,
    input_desc.M,
    input_desc.rank,
    input_desc.uniqueRanks().size());
}

void qnFit(raft::handle_t& handle,
           std::vector<Matrix::Data<float>*>& input_data,
           Matrix::PartDescriptor& input_desc,
           std::vector<Matrix::Data<float>*>& labels,
           float* coef,
           const qn_params& pams,
           bool X_col_major,
           float* f,
           int* num_iters) 
{
  qnFit_impl(
    handle,
    input_data,
    input_desc,
    labels,
    coef,
    pams,
    X_col_major,
    f,
    num_iters
  );

}

/*
void qnFit(const raft::handle_t &handle, 
           const qn_params& pams,
           float* X,
           bool X_col_major,
           float *y,
           int N,
           int D,
           int C,
           float* w0,
           float* f,
           int* num_iters,
           int n_samples,
           int rank,
           int n_ranks)
{
  qnFit_impl<float, int>(
    handle,
    pams,
    X,
    X_col_major,
    y,
    N,
    D,
    C,
    w0,
    f,
    num_iters,
    n_samples,
    rank,
    n_ranks);
}
*/

};  // namespace OPG
};  // namespace GLM
};  // namespace ML

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

#pragma once

#include <glm/qn/simple_mat/dense.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/operators.hpp>

#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/multiply.cuh>
#include <raft/linalg/sqrt.cuh>
#include <raft/matrix/math.hpp>
#include <raft/stats/stddev.cuh>
#include <raft/stats/sum.cuh>

#include <cuml/common/logger.hpp>

namespace ML {
namespace GLM {
namespace opg {

template <typename T>
void mean_stddev(const raft::handle_t& handle,
                 const SimpleDenseMat<T>& X,
                 int n_samples,
                 T* mean_vector,
                 T* stddev_vector)
{
  const T* input_data = X.data;
  int D               = X.n;
  int num_rows        = X.m;
  bool col_major      = (X.ord == COL_MAJOR);
  auto stream         = handle.get_stream();
  auto& comm          = handle.get_comms();

  raft::stats::sum(mean_vector, input_data, D, num_rows, !col_major, stream);
  T weight = T(1) / T(n_samples);
  raft::linalg::multiplyScalar(mean_vector, mean_vector, weight, D, stream);
  comm.allreduce(mean_vector, mean_vector, D, raft::comms::op_t::SUM, stream);
  comm.sync_stream(stream);

  raft::stats::vars(stddev_vector, input_data, mean_vector, D, num_rows, false, !col_major, stream);
  weight = T(1) * num_rows / T(n_samples);
  raft::linalg::multiplyScalar(stddev_vector, stddev_vector, weight, D, stream);
  comm.allreduce(stddev_vector, stddev_vector, D, raft::comms::op_t::SUM, stream);
  comm.sync_stream(stream);

  raft::linalg::sqrt(stddev_vector, stddev_vector, D, handle.get_stream());
}

template <typename T>
void scale_model(const raft::handle_t& handle,
                 T* coef,
                 int n_targets,
                 int D,
                 bool has_bias,
                 const SimpleVec<T>& mean,
                 const SimpleVec<T>& stddev)
{
  // adapt coefficients and intercept to avoid actual data standardization
  SimpleDenseMat<T> W(coef, n_targets, D + has_bias);
  SimpleDenseMat<T> Wweights;
  col_slice(W, Wweights, 0, D);

  auto div_stddev = [] __device__(const T a, const T b) {
    if (b == 0.0) return (T)0.0;
    return a / b;
  };
  raft::linalg::matrixVectorOp(Wweights.data,
                               Wweights.data,
                               stddev.data,
                               Wweights.n,
                               Wweights.m,
                               false,
                               true,
                               div_stddev,
                               handle.get_stream());

  if (has_bias) {
    SimpleVec<T> Wbias;

    col_ref(W, Wbias, D);

    Wbias.assign_gemv(handle, -1, Wweights, false, mean, 1, handle.get_stream());
  }
}

template <typename T>
void adapt_gradient_linearBwd(const raft::handle_t& handle,
                              SimpleDenseMat<T>& G,
                              const SimpleDenseMat<T>& dZ,
                              bool has_bias,
                              int n_samples,
                              const SimpleVec<T>& mean,
                              const SimpleVec<T>& stddev)
{
  auto stream   = handle.get_stream();
  int D         = mean.len;
  int n_targets = dZ.m;
  auto& comm    = handle.get_comms();

  // calculate scaled mean: mean / stddev
  rmm::device_uvector<T> scaledMean(D, stream);
  raft::linalg::binaryOp(
    scaledMean.data(), mean.data, stddev.data, D, raft::div_checkzero_op(), stream);

  auto log_scaledMean = raft::arr2Str(scaledMean.data(), D, "", stream, 8);
  CUML_LOG_DEBUG("adapt_gradient::log_scaledMean is %s", log_scaledMean.c_str());

  // calculate adaption
  rmm::device_uvector<T> avgDz(n_targets, stream);
  raft::stats::sum(
    avgDz.data(), dZ.data, n_targets, dZ.n, true, stream);  // sum performs on each row of dZ
  comm.allreduce(avgDz.data(), avgDz.data(), n_targets, raft::comms::op_t::SUM, stream);
  // T scalar = T(n_samples);
  // raft::linalg::divideScalar(avgDz.data(), avgDz.data(), scalar, n_targets, stream);

  auto log_avgDz = raft::arr2Str(avgDz.data(), n_targets, "", stream, 8);
  CUML_LOG_DEBUG("adapt_gradient::log_avgDz is %s", log_avgDz.c_str());

  SimpleDenseMat<T> Gweights;
  col_slice(G, Gweights, 0, D);
  SimpleDenseMat<T> avgDzMat(avgDz.data(), n_targets, 1);
  SimpleDenseMat<T> scaledMeanMat(scaledMean.data(), 1, D);
  Gweights.assign_gemm(
    handle, -1.0 / n_samples, avgDzMat, false, scaledMeanMat, false, 1.0, stream);

  auto log_Gweights = raft::arr2Str(Gweights.data, D * n_targets, "", stream, 8);
  CUML_LOG_DEBUG("adapt_gradient::log_Gweights is %s", log_Gweights.c_str());
}

};  // namespace opg
};  // namespace GLM
};  // namespace ML
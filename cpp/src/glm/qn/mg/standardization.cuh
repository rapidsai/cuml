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

#pragma once

#include <glm/qn/simple_mat/dense.hpp>
#include <raft/core/comms.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/operators.hpp>

#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/divide.cuh>
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
  weight = n_samples < 1 ? T(0) : T(1) * num_rows / T(n_samples - 1);
  raft::linalg::multiplyScalar(stddev_vector, stddev_vector, weight, D, stream);
  comm.allreduce(stddev_vector, stddev_vector, D, raft::comms::op_t::SUM, stream);
  comm.sync_stream(stream);

  raft::linalg::sqrt(stddev_vector, stddev_vector, D, handle.get_stream());
}

template <typename T>
void adapt_model_for_linearFwd(const raft::handle_t& handle,
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
void adapt_gradient_for_linearBwd(const raft::handle_t& handle,
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

  // calculate std_inv
  rmm::device_uvector<T> std_inv(D, stream);
  auto inverse_op = [] __device__(const T value) {
    return value == T(0.0) ? value : T(1.0) / value;
  };
  raft::linalg::unaryOp(std_inv.data(), stddev.data, D, inverse_op, stream);

  // auto log_std_inv= raft::arr2Str(std_inv.data(), D, "", stream, 8);
  // CUML_LOG_DEBUG("log_std_inv is %s", log_std_inv.c_str());

  // scale coefficients
  SimpleDenseMat<T> Gweights;
  col_slice(G, Gweights, 0, D);

  // auto log_before = raft::arr2Str(Gweights.data, Gweights.m * D, "", stream, 8);
  // CUML_LOG_DEBUG("Gweights before is %s", log_before.c_str());

  raft::matrix::matrixVectorBinaryMult(
    Gweights.data, std_inv.data(), Gweights.m, D, false, true, stream);

  // auto log_Gweights = raft::arr2Str(Gweights.data, Gweights.m * D, "", stream, 8);
  // CUML_LOG_DEBUG("Gweights after is %s", log_Gweights.c_str());

  if (has_bias) {
    SimpleVec<T> Gbias;
    col_ref(G, Gbias, D);

    // calculate scaledMean
    raft::linalg::binaryOp(std_inv.data(), std_inv.data(), mean.data, D, raft::mul_op(), stream);
    // auto log_scaledMean = raft::arr2Str(std_inv.data(),  D, "", stream, 8);
    // CUML_LOG_DEBUG("log_scaledMean is %s", log_scaledMean.c_str());

    SimpleDenseMat<T> Gbias_transpose_mat(Gbias.data, Gbias.m, 1);
    SimpleDenseMat<T> scaled_mean_mat(std_inv.data(), 1, D);

    Gweights.assign_gemm(
      handle, -1.0, Gbias_transpose_mat, false, scaled_mean_mat, false, 1.0, stream);
    // auto log_Gweights_adapted = raft::arr2Str(Gweights.data, Gweights.m * D, "", stream, 8);
    // CUML_LOG_DEBUG("log_Gweights_adapted is %s", log_Gweights_adapted.c_str());
  }
}

};  // namespace opg
};  // namespace GLM
};  // namespace ML
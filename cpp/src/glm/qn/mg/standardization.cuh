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

#include <cuml/common/logger.hpp>

#include <raft/core/comms.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/handle.hpp>
#include <raft/core/operators.hpp>
#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/divide.cuh>
#include <raft/linalg/multiply.cuh>
#include <raft/linalg/sqrt.cuh>
#include <raft/matrix/init.cuh>
#include <raft/matrix/math.hpp>
#include <raft/sparse/op/row_op.cuh>
#include <raft/stats/stddev.cuh>
#include <raft/stats/sum.cuh>
#include <raft/util/cudart_utils.hpp>

#include <cuda_runtime_api.h>

#include <glm/qn/simple_mat/dense.hpp>
#include <glm/qn/simple_mat/sparse.hpp>

#include <vector>

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
void mean_stddev(const raft::handle_t& handle,
                 const SimpleSparseMat<T>& X,
                 int n_samples,
                 T* mean_vector,
                 T* stddev_vector)
{
  ML::Logger::get().setLevel(6);
  CUML_LOG_DEBUG(
    "sparkdebug mean_vector addr: %p, stddev_vector addr: %p", mean_vector, stddev_vector);

  int D        = X.n;
  int num_rows = X.m;
  auto stream  = handle.get_stream();
  auto& comm   = handle.get_comms();
  SimpleDenseMat<T> mean_mat(mean_vector, 1, D);

  rmm::device_uvector<T> ones(num_rows, stream);
  auto ones_view = raft::make_device_vector_view(ones.data(), num_rows);
  raft::matrix::fill(handle, ones_view, T(1.0));
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  SimpleDenseMat<T> ones_mat(ones.data(), 1, num_rows);

  // calculate stdev.S
  SimpleDenseMat<T> stddev_mat(stddev_vector, 1, D);

  ML::Logger::get().setLevel(6);
  rmm::device_uvector<T> values_copy(X.nnz, stream);
  auto copied_size = X.nnz * sizeof(T);
  // raft::copy(values_copy.data(), X.values, X.nnz, stream);
  RAFT_CUDA_TRY(
    cudaMemcpyAsync(values_copy.data(), X.values, copied_size, cudaMemcpyDeviceToDevice, stream));
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  handle.sync_stream();
  CUML_LOG_DEBUG("sparkdebug finished copying X.values with X.nnz: %d", X.nnz);

  auto square_op = [] __device__(const T a) { return a * a; };
  raft::linalg::unaryOp(values_copy.data(), values_copy.data(), X.nnz, square_op, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  CUML_LOG_DEBUG(
    "sparkdebug mean_vector addr: %p, stddev_vector addr: %p", mean_vector, stddev_vector);
  auto X_square_tmp = SimpleSparseMat<T>(values_copy.data(), X.cols, X.row_ids, X.nnz, num_rows, D);
  X_square_tmp.gemmb(handle, T(1.), ones_mat, false, false, T(0.), stddev_mat, stream);

  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  CUML_LOG_DEBUG("sparkdebug finished X.gemmb with X.nnz: %d, stddev: %p", X.nnz, stddev_vector);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  T weight = n_samples < 1 ? T(0) : T(1) / T(n_samples - 1);
  raft::linalg::multiplyScalar(stddev_vector, stddev_vector, weight, D, stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  comm.allreduce(stddev_vector, stddev_vector, D, raft::comms::op_t::SUM, stream);
  comm.sync_stream(stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
  auto log_stddev_vector = raft::arr2Str(stddev_vector, D, "", stream);
  CUML_LOG_DEBUG("sparkdebug finished allreduce stddev_Vector: %s", log_stddev_vector.c_str());

  // calculate mean
  X.gemmb(handle, 1., ones_mat, false, false, 0., mean_mat, stream);
  weight = T(1) / T(n_samples);
  raft::linalg::multiplyScalar(mean_vector, mean_vector, weight, D, stream);
  comm.allreduce(mean_vector, mean_vector, D, raft::comms::op_t::SUM, stream);
  comm.sync_stream(stream);
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream));

  weight          = T(n_samples) / T(n_samples - 1);
  auto submean_op = [weight] __device__(const T a, const T b) { return a - b * b * weight; };
  raft::linalg::binaryOp(stddev_vector, stddev_vector, mean_vector, D, submean_op, stream);

  raft::linalg::sqrt(stddev_vector, stddev_vector, D, handle.get_stream());
}

struct inverse_op {
  template <typename T>
  constexpr RAFT_INLINE_FUNCTION auto operator()(const T& a) const
  {
    return a == T(0.0) ? a : T(1.0) / a;
  }
};

template <typename T>
struct Standardizer {
  SimpleVec<T> mean;
  SimpleVec<T> std;
  SimpleVec<T> std_inv;
  SimpleVec<T> scaled_mean;

  Standardizer(const raft::handle_t& handle,
               const SimpleDenseMat<T>& X,
               int n_samples,
               rmm::device_uvector<T>& mean_std_buff)
  {
    int D = X.n;
    ASSERT(mean_std_buff.size() == 4 * D, "buff size must be four times the dimension");

    auto stream = handle.get_stream();

    mean.reset(mean_std_buff.data(), D);
    std.reset(mean_std_buff.data() + D, D);
    std_inv.reset(mean_std_buff.data() + 2 * D, D);
    scaled_mean.reset(mean_std_buff.data() + 3 * D, D);

    mean_stddev(handle, X, n_samples, mean.data, std.data);

    raft::linalg::unaryOp(std_inv.data, std.data, D, inverse_op(), stream);

    // scale mean by the standard deviation
    raft::linalg::binaryOp(scaled_mean.data, std_inv.data, mean.data, D, raft::mul_op(), stream);
  }

  Standardizer(const raft::handle_t& handle,
               const SimpleSparseMat<T>& X,
               int n_samples,
               rmm::device_uvector<T>& mean_std_buff,
               size_t vec_size)
  {
    int D = X.n;
    ASSERT(mean_std_buff.size() == 4 * vec_size, "buff size must be four times the aligned size");

    auto stream = handle.get_stream();

    T* p_ws = mean_std_buff.data();

    mean.reset(p_ws, D);
    p_ws += vec_size;

    std.reset(p_ws, D);
    p_ws += vec_size;

    std_inv.reset(p_ws, D);
    p_ws += vec_size;

    scaled_mean.reset(p_ws, D);

    mean_stddev(handle, X, n_samples, mean.data, std.data);
    raft::linalg::unaryOp(std_inv.data, std.data, D, inverse_op(), stream);

    // scale mean by the standard deviation
    raft::linalg::binaryOp(scaled_mean.data, std_inv.data, mean.data, D, raft::mul_op(), stream);
  }

  void adapt_model_for_linearFwd(
    const raft::handle_t& handle, T* coef, int n_targets, int D, bool has_bias) const
  {
    ASSERT(D == mean.len, "dimension mismatches");

    // adapt coefficients and intercept to avoid actual data standardization
    SimpleDenseMat<T> W(coef, n_targets, D + has_bias);
    SimpleDenseMat<T> Wweights;
    col_slice(W, Wweights, 0, D);

    auto mul_lambda = [] __device__(const T a, const T b) { return a * b; };
    raft::linalg::matrixVectorOp(Wweights.data,
                                 Wweights.data,
                                 std_inv.data,
                                 Wweights.n,
                                 Wweights.m,
                                 false,
                                 true,
                                 mul_lambda,
                                 handle.get_stream());

    if (has_bias) {
      SimpleVec<T> Wbias;

      col_ref(W, Wbias, D);

      Wbias.assign_gemv(handle, -1, Wweights, false, mean, 1, handle.get_stream());
    }
  }

  void adapt_gradient_for_linearBwd(const raft::handle_t& handle,
                                    SimpleDenseMat<T>& G,
                                    const SimpleDenseMat<T>& dZ,
                                    bool has_bias,
                                    int n_samples) const
  {
    auto stream   = handle.get_stream();
    int D         = mean.len;
    int n_targets = dZ.m;
    auto& comm    = handle.get_comms();

    // scale coefficients
    SimpleDenseMat<T> Gweights;
    col_slice(G, Gweights, 0, D);

    raft::matrix::matrixVectorBinaryMult(
      Gweights.data, std_inv.data, Gweights.m, D, false, true, stream);

    if (has_bias) {
      SimpleVec<T> Gbias;
      col_ref(G, Gbias, D);

      SimpleDenseMat<T> Gbias_transpose_mat(Gbias.data, Gbias.m, 1);
      SimpleDenseMat<T> scaled_mean_mat(scaled_mean.data, 1, D);

      Gweights.assign_gemm(
        handle, -1.0, Gbias_transpose_mat, false, scaled_mean_mat, false, 1.0, stream);
    }
  }
};

};  // namespace opg
};  // namespace GLM
};  // namespace ML
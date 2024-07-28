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
#include <raft/core/handle.hpp>
#include <raft/core/operators.hpp>
#include <raft/linalg/binary_op.cuh>
#include <raft/linalg/divide.cuh>
#include <raft/linalg/multiply.cuh>
#include <raft/linalg/sqrt.cuh>
#include <raft/linalg/subtract.cuh>
#include <raft/matrix/math.hpp>
#include <raft/sparse/op/row_op.cuh>
#include <raft/stats/stddev.cuh>
#include <raft/stats/sum.cuh>

#include <glm/qn/simple_mat/dense.hpp>
#include <glm/qn/simple_mat/sparse.hpp>

#include <vector>

namespace ML {
namespace GLM {
namespace opg {

template <typename T>
void mean_stddev(const raft::handle_t& handle,
                 const SimpleDenseMat<T>& X,
                 size_t n_samples,
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

  // avoid negative variance that is due to precision loss of floating point arithmetic
  weight         = n_samples < 1 ? T(0) : T(1) / T(n_samples - 1);
  weight         = n_samples * weight;
  auto no_neg_op = [weight] __device__(const T a, const T b) -> T {
    if (a >= 0) return a;

    return a + weight * b * b;
  };

  raft::linalg::binaryOp(stddev_vector, stddev_vector, mean_vector, D, no_neg_op, stream);

  raft::linalg::sqrt(stddev_vector, stddev_vector, D, handle.get_stream());
}

template <typename T, typename I = int>
SimpleSparseMat<T, I> get_sub_mat(const raft::handle_t& handle,
                                  SimpleSparseMat<T, I> mat,
                                  int start,
                                  int end,
                                  rmm::device_uvector<I>& buff_row_ids)
{
  end         = end <= mat.m ? end : mat.m;
  int n_rows  = end - start;
  int n_cols  = mat.n;
  auto stream = handle.get_stream();

  RAFT_EXPECTS(start < end, "start index must be smaller than end index");
  RAFT_EXPECTS(buff_row_ids.size() >= n_rows + 1,
               "the size of buff_row_ids should be at least end - start + 1");
  raft::copy(buff_row_ids.data(), mat.row_ids + start, n_rows + 1, stream);

  I idx;
  raft::copy(&idx, buff_row_ids.data(), 1, stream);
  raft::resource::sync_stream(handle);

  auto subtract_op = [idx] __device__(const I a) { return a - idx; };
  raft::linalg::unaryOp(buff_row_ids.data(), buff_row_ids.data(), n_rows + 1, subtract_op, stream);

  I nnz;
  raft::copy(&nnz, buff_row_ids.data() + n_rows, 1, stream);
  raft::resource::sync_stream(handle);

  SimpleSparseMat<T, I> res(
    mat.values + idx, mat.cols + idx, buff_row_ids.data(), nnz, n_rows, n_cols);
  return res;
}

template <typename T, typename I = int>
void mean(const raft::handle_t& handle,
          const SimpleSparseMat<T, I>& X,
          size_t n_samples,
          T* mean_vector)
{
  int D        = X.n;
  int num_rows = X.m;
  auto stream  = handle.get_stream();
  auto& comm   = handle.get_comms();

  int chunk_size = 500000;  // split matrix by rows for better numeric precision
  rmm::device_uvector<I> buff_row_ids(chunk_size + 1, stream);

  rmm::device_uvector<T> ones(chunk_size, stream);
  SimpleVec<T> ones_vec(ones.data(), chunk_size);
  ones_vec.fill(1.0, stream);

  rmm::device_uvector<T> buff_D(D, stream);
  SimpleDenseMat<T> buff_D_mat(buff_D.data(), 1, D);

  // calculate mean
  SimpleDenseMat<T> mean_mat(mean_vector, 1, D);
  mean_mat.fill(0., stream);

  for (int i = 0; i < X.m; i += chunk_size) {
    // get X[i:i + chunk_size]
    SimpleSparseMat<T, I> X_sub = get_sub_mat(handle, X, i, i + chunk_size, buff_row_ids);
    SimpleDenseMat<T> ones_mat(ones.data(), 1, X_sub.m);

    X_sub.gemmb(handle, 1., ones_mat, false, false, 0., buff_D_mat, stream);
    raft::linalg::binaryOp(mean_vector, mean_vector, buff_D_mat.data, D, raft::add_op(), stream);
  }

  T weight = T(1) / T(n_samples);
  raft::linalg::multiplyScalar(mean_vector, mean_vector, weight, D, stream);
  comm.allreduce(mean_vector, mean_vector, D, raft::comms::op_t::SUM, stream);
  comm.sync_stream(stream);
}

template <typename T, typename I = int>
void mean_stddev(const raft::handle_t& handle,
                 const SimpleSparseMat<T, I>& X,
                 size_t n_samples,
                 T* mean_vector,
                 T* stddev_vector)
{
  auto stream = handle.get_stream();
  int D       = X.n;
  mean(handle, X, n_samples, mean_vector);

  // calculate stdev.S
  rmm::device_uvector<T> X_values_squared(X.nnz, stream);
  raft::copy(X_values_squared.data(), X.values, X.nnz, stream);
  auto square_op = [] __device__(const T a) { return a * a; };
  raft::linalg::unaryOp(X_values_squared.data(), X_values_squared.data(), X.nnz, square_op, stream);

  auto X_squared =
    SimpleSparseMat<T, I>(X_values_squared.data(), X.cols, X.row_ids, X.nnz, X.m, X.n);

  mean(handle, X_squared, n_samples, stddev_vector);

  T weight               = n_samples / T(n_samples - 1);
  auto submean_no_neg_op = [weight] __device__(const T a, const T b) -> T {
    T res = weight * (a - b * b);
    if (res < 0) {
      // return sum(x^2) / (n - 1) if negative variance (due to precision loss of floating point
      // arithmetic)
      res = weight * a;
    }
    return res;
  };
  raft::linalg::binaryOp(stddev_vector, stddev_vector, mean_vector, X.n, submean_no_neg_op, stream);

  raft::linalg::sqrt(stddev_vector, stddev_vector, X.n, handle.get_stream());
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
               size_t n_samples,
               rmm::device_uvector<T>& mean_std_buff)
  {
    int D = X.n;
    ASSERT(mean_std_buff.size() == 4 * D, "mean_std_buff size must be four times the dimension");

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

  template <typename I = int>
  Standardizer(const raft::handle_t& handle,
               const SimpleSparseMat<T, I>& X,
               size_t n_samples,
               rmm::device_uvector<T>& mean_std_buff,
               size_t vec_size)
  {
    int D = X.n;
    ASSERT(mean_std_buff.size() == 4 * vec_size,
           "mean_std_buff size must be four times the aligned size");

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
                                    bool has_bias) const
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
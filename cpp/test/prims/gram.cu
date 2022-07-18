/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include "test_utils.h"
#include <cuml/matrix/kernelparams.h>
#include <gtest/gtest.h>
#include <iostream>
#include <matrix/grammatrix.cuh>
#include <matrix/kernelfactory.cuh>
#include <memory>
#include <raft/cuda_utils.cuh>
#include <raft/cudart_utils.h>
#include <raft/random/rng.hpp>
#include <rmm/device_uvector.hpp>

namespace MLCommon {
namespace Matrix {

// Get the offset of element [i,k].
HDI int get_offset(int i, int k, int ld, bool is_row_major)
{
  return is_row_major ? i * ld + k : i + k * ld;
}

struct GramMatrixInputs {
  int n1;      // feature vectors in matrix 1
  int n2;      // featuer vectors in matrix 2
  int n_cols;  // number of elements in a feature vector
  bool is_row_major;
  KernelParams kernel;
  int ld1;
  int ld2;
  int ld_out;
  // We will generate random input using the dimensions given here.
  // The reference output is calculated by a custom kernel.
};

std::ostream& operator<<(std::ostream& os, const GramMatrixInputs& p)
{
  std::vector<std::string> kernel_names{"linear", "poly", "rbf", "tanh"};
  os << "/" << p.n1 << "x" << p.n2 << "x" << p.n_cols << "/"
     << (p.is_row_major ? "RowMajor/" : "ColMajor/") << kernel_names[p.kernel.kernel] << "/ld_"
     << p.ld1 << "x" << p.ld2 << "x" << p.ld_out;
  return os;
}

const std::vector<GramMatrixInputs> inputs = {
  {42, 137, 2, false, {KernelType::LINEAR}},
  {42, 137, 2, true, {KernelType::LINEAR}},
  {42, 137, 2, false, {KernelType::LINEAR}, 64, 179, 181},
  {42, 137, 2, true, {KernelType::LINEAR}, 64, 179, 181},
  {137, 42, 2, false, {KernelType::POLYNOMIAL, 2, 0.5, 2.4}},
  {137, 42, 2, true, {KernelType::POLYNOMIAL, 2, 0.5, 2.4}},
  {137, 42, 2, false, {KernelType::POLYNOMIAL, 2, 0.5, 2.4}, 159, 73, 144},
  {137, 42, 2, true, {KernelType::POLYNOMIAL, 2, 0.5, 2.4}, 159, 73, 144},
  {42, 137, 2, false, {KernelType::TANH, 0, 0.5, 2.4}},
  {42, 137, 2, true, {KernelType::TANH, 0, 0.5, 2.4}},
  {42, 137, 2, false, {KernelType::TANH, 0, 0.5, 2.4}, 64, 155, 49},
  {42, 137, 2, true, {KernelType::TANH, 0, 0.5, 2.4}, 64, 155, 143},
  {3, 4, 2, false, {KernelType::RBF, 0, 0.5}},
  {42, 137, 2, false, {KernelType::RBF, 0, 0.5}},
  {42, 137, 2, true, {KernelType::RBF, 0, 0.5}},
  // Distance kernel does not support LD parameter yet.
  //{42, 137, 2, false, {KernelType::RBF, 0, 0.5}, 64, 155, 49},
  // {42, 137, 2, true, {KernelType::RBF, 0, 0.5}, 64, 155, 143},
};

template <typename math_t>
class GramMatrixTest : public ::testing::TestWithParam<GramMatrixInputs> {
 protected:
  GramMatrixTest()
    : params(GetParam()), stream(0), x1(0, stream), x2(0, stream), gram(0, stream), gram_host(0)
  {
    RAFT_CUDA_TRY(cudaStreamCreate(&stream));

    if (params.ld1 == 0) { params.ld1 = params.is_row_major ? params.n_cols : params.n1; }
    if (params.ld2 == 0) { params.ld2 = params.is_row_major ? params.n_cols : params.n2; }
    if (params.ld_out == 0) { params.ld_out = params.is_row_major ? params.n2 : params.n1; }
    // Derive the size of the ouptut from the offset of the last element.
    size_t size = get_offset(params.n1 - 1, params.n_cols - 1, params.ld1, params.is_row_major) + 1;
    x1.resize(size, stream);
    size = get_offset(params.n2 - 1, params.n_cols - 1, params.ld2, params.is_row_major) + 1;
    x2.resize(size, stream);
    size = get_offset(params.n1 - 1, params.n2 - 1, params.ld_out, params.is_row_major) + 1;

    gram.resize(size, stream);
    RAFT_CUDA_TRY(cudaMemsetAsync(gram.data(), 0, gram.size() * sizeof(math_t), stream));
    gram_host.resize(gram.size());
    std::fill(gram_host.begin(), gram_host.end(), 0);

    raft::random::Rng r(42137ULL);
    r.uniform(x1.data(), x1.size(), math_t(0), math_t(1), stream);
    r.uniform(x2.data(), x2.size(), math_t(0), math_t(1), stream);
  }

  ~GramMatrixTest() override { RAFT_CUDA_TRY_NO_THROW(cudaStreamDestroy(stream)); }

  // Calculate the Gram matrix on the host.
  void naiveKernel()
  {
    std::vector<math_t> x1_host(x1.size());
    raft::update_host(x1_host.data(), x1.data(), x1.size(), stream);
    std::vector<math_t> x2_host(x2.size());
    raft::update_host(x2_host.data(), x2.data(), x2.size(), stream);
    handle.sync_stream(stream);

    for (int i = 0; i < params.n1; i++) {
      for (int j = 0; j < params.n2; j++) {
        float d = 0;
        for (int k = 0; k < params.n_cols; k++) {
          if (params.kernel.kernel == KernelType::RBF) {
            math_t diff = x1_host[get_offset(i, k, params.ld1, params.is_row_major)] -
                          x2_host[get_offset(j, k, params.ld2, params.is_row_major)];
            d += diff * diff;
          } else {
            d += x1_host[get_offset(i, k, params.ld1, params.is_row_major)] *
                 x2_host[get_offset(j, k, params.ld2, params.is_row_major)];
          }
        }
        int idx  = get_offset(i, j, params.ld_out, params.is_row_major);
        math_t v = 0;
        switch (params.kernel.kernel) {
          case (KernelType::LINEAR): gram_host[idx] = d; break;
          case (KernelType::POLYNOMIAL):
            v              = params.kernel.gamma * d + params.kernel.coef0;
            gram_host[idx] = std::pow(v, params.kernel.degree);
            break;
          case (KernelType::TANH):
            gram_host[idx] = std::tanh(params.kernel.gamma * d + params.kernel.coef0);
            break;
          case (KernelType::RBF): gram_host[idx] = exp(-params.kernel.gamma * d); break;
        }
      }
    }
  }

  void runTest()
  {
    std::unique_ptr<GramMatrixBase<math_t>> kernel = std::unique_ptr<GramMatrixBase<math_t>>(
      KernelFactory<math_t>::create(params.kernel, handle.get_cublas_handle()));

    kernel->evaluate(x1.data(),
                     params.n1,
                     params.n_cols,
                     x2.data(),
                     params.n2,
                     gram.data(),
                     params.is_row_major,
                     stream,
                     params.ld1,
                     params.ld2,
                     params.ld_out);
    naiveKernel();
    ASSERT_TRUE(raft::devArrMatchHost(
      gram_host.data(), gram.data(), gram.size(), raft::CompareApprox<math_t>(1e-6f)));
  }

  raft::handle_t handle;
  cudaStream_t stream = 0;
  GramMatrixInputs params;

  rmm::device_uvector<math_t> x1;
  rmm::device_uvector<math_t> x2;
  rmm::device_uvector<math_t> gram;
  std::vector<math_t> gram_host;
};

typedef GramMatrixTest<float> GramMatrixTestFloat;
typedef GramMatrixTest<double> GramMatrixTestDouble;

TEST_P(GramMatrixTestFloat, Gram) { runTest(); }

INSTANTIATE_TEST_SUITE_P(GramMatrixTests, GramMatrixTestFloat, ::testing::ValuesIn(inputs));
};  // end namespace Matrix
};  // end namespace MLCommon

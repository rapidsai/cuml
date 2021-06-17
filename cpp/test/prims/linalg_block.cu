/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

#include <gtest/gtest.h>

#include <random>
#include <vector>

#include <raft/cudart_utils.h>
#include <raft/cuda_utils.cuh>
#include <raft/handle.hpp>
#include <raft/mr/device/allocator.hpp>

#include <test_utils.h>

#include <cuml/common/device_buffer.hpp>
#include <cuml/common/logger.hpp>

#include <linalg/block.cuh>

namespace MLCommon {
namespace LinAlg {

using namespace std;

/* GEMM */

template <typename T>
struct BlockGemmInputs {
  int m, k, n;
  bool transa, transb;
  int batch_size;
  T eps;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const BlockGemmInputs<T>& dims) {
  return os;
}

template <typename Policy, typename T>
__global__ void block_gemm_test_kernel(bool transa, bool transb, int m, int n,
                                       int k, T alpha, const T* a, const T* b,
                                       T* c) {
  __shared__ MLCommon::LinAlg::GemmStorage<Policy, T> gemm_storage;

  _block_gemm<Policy>(transa, transb, m, n, k, alpha, a + m * k * blockIdx.x,
                      b + k * n * blockIdx.x, c + m * n * blockIdx.x,
                      gemm_storage);
}

template <typename T>
class BlockGemmTest : public ::testing::TestWithParam<BlockGemmInputs<T>> {
 protected:
  void basicTest() {
    raft::handle_t handle;

    params = ::testing::TestWithParam<BlockGemmInputs<T>>::GetParam();

    device_buffer<T> a(handle.get_device_allocator(), handle.get_stream(),
                       params.m * params.k * params.batch_size);
    device_buffer<T> b(handle.get_device_allocator(), handle.get_stream(),
                       params.k * params.n * params.batch_size);
    device_buffer<T> c(handle.get_device_allocator(), handle.get_stream(),
                       params.m * params.n * params.batch_size);

    std::vector<T> h_a(params.m * params.k * params.batch_size);
    std::vector<T> h_b(params.k * params.n * params.batch_size);
    std::vector<T> h_c_ref(params.m * params.n * params.batch_size);
    std::vector<T> h_c_dev(params.m * params.n * params.batch_size);

    /* Generate random data */
    std::default_random_engine generator(params.seed);
    std::uniform_real_distribution<double> distribution(-2.0, 2.0);
    for (int i = 0; i < params.m * params.k * params.batch_size; i++)
      h_a[i] = distribution(generator);
    for (int i = 0; i < params.k * params.n * params.batch_size; i++)
      h_b[i] = distribution(generator);
    T alpha = distribution(generator);

    /* Copy to device */
    raft::copy(a.data(), h_a.data(), params.m * params.k * params.batch_size,
               handle.get_stream());
    raft::copy(b.data(), h_b.data(), params.k * params.n * params.batch_size,
               handle.get_stream());

    /* Compute reference results */
    for (int bid = 0; bid < params.batch_size; bid++) {
      for (int i = 0; i < params.m; i++) {
        for (int j = 0; j < params.n; j++) {
          T acc = (T)0;
          for (int h = 0; h < params.k; h++) {
            T _a = params.transa
                     ? h_a[bid * params.m * params.k + i * params.k + h]
                     : h_a[bid * params.m * params.k + h * params.m + i];
            T _b = params.transb
                     ? h_b[bid * params.k * params.n + h * params.n + j]
                     : h_b[bid * params.k * params.n + j * params.k + h];
            acc += _a * _b;
          }

          h_c_ref[bid * params.m * params.n + j * params.m + i] = alpha * acc;
        }
      }
    }

    /* Compute using tested prims */
    using Policy = BlockGemmPolicy<32, 2, 2, 16, 16>;
    block_gemm_test_kernel<Policy>
      <<<params.batch_size, Policy::BlockSize, 0, handle.get_stream()>>>(
        params.transa, params.transb, params.m, params.n, params.k, alpha,
        a.data(), b.data(), c.data());

    /* Copy results to host */
    raft::copy(h_c_dev.data(), c.data(),
               params.m * params.n * params.batch_size, handle.get_stream());
    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    /* Count errors */
    errors = 0;
    for (int i = 0; i < params.m * params.n * params.batch_size; i++) {
      T diff = abs(h_c_ref[i] - h_c_dev[i]);
      if (std::isnan(diff) || diff > params.eps) errors++;
    }
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  BlockGemmInputs<T> params;

  int errors;
};

const std::vector<BlockGemmInputs<float>> gemm_inputsf = {
  {42, 42, 42, false, false, 20, 1e-4, 12345U},
  {65, 10, 20, false, true, 50, 1e-4, 12345U},
  {5, 80, 31, true, false, 100, 1e-4, 12345U}};

const std::vector<BlockGemmInputs<double>> gemm_inputsd = {
  {42, 42, 42, false, false, 20, 1e-4, 12345U},
  {65, 10, 20, false, true, 50, 1e-4, 12345U},
  {5, 80, 31, true, false, 100, 1e-4, 12345U}};

typedef BlockGemmTest<float> BlockGemmTestF;
TEST_P(BlockGemmTestF, Result) { ASSERT_EQ(errors, 0); }

typedef BlockGemmTest<double> BlockGemmTestD;
TEST_P(BlockGemmTestD, Result) { ASSERT_EQ(errors, 0); }

INSTANTIATE_TEST_CASE_P(BlockGemmTests, BlockGemmTestF,
                        ::testing::ValuesIn(gemm_inputsf));

INSTANTIATE_TEST_CASE_P(BlockGemmTests, BlockGemmTestD,
                        ::testing::ValuesIn(gemm_inputsd));

/* GEMV */

template <typename T>
struct BlockGemvInputs {
  int m, n;
  int batch_size;
  T eps;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const BlockGemvInputs<T>& dims) {
  return os;
}

template <typename Policy, typename T>
__global__ void block_gemv_test_kernel(int m, int n, T alpha, const T* a,
                                       const T* x, T* y) {
  __shared__ MLCommon::LinAlg::GemvStorage<Policy, T> gemv_storage;

  extern __shared__ char dyna_shared_mem[];
  T* shared_vec = (T*)dyna_shared_mem;

  _block_gemv<Policy>(m, n, alpha, a + m * n * blockIdx.x, x + n * blockIdx.x,
                      y + m * blockIdx.x, gemv_storage, shared_vec);
}

template <typename T>
class BlockGemvTest : public ::testing::TestWithParam<BlockGemvInputs<T>> {
 protected:
  void basicTest() {
    raft::handle_t handle;

    params = ::testing::TestWithParam<BlockGemvInputs<T>>::GetParam();

    device_buffer<T> a(handle.get_device_allocator(), handle.get_stream(),
                       params.m * params.n * params.batch_size);
    device_buffer<T> x(handle.get_device_allocator(), handle.get_stream(),
                       params.n * params.batch_size);
    device_buffer<T> y(handle.get_device_allocator(), handle.get_stream(),
                       params.m * params.batch_size);

    std::vector<T> h_a(params.m * params.n * params.batch_size);
    std::vector<T> h_x(params.n * params.batch_size);
    std::vector<T> h_y_ref(params.m * params.batch_size);
    std::vector<T> h_y_dev(params.m * params.batch_size);

    /* Generate random data */
    std::default_random_engine generator(params.seed);
    std::uniform_real_distribution<double> distribution(-2.0, 2.0);
    for (int i = 0; i < params.m * params.n * params.batch_size; i++)
      h_a[i] = distribution(generator);
    for (int i = 0; i < params.n * params.batch_size; i++)
      h_x[i] = distribution(generator);
    T alpha = distribution(generator);

    /* Copy to device */
    raft::copy(a.data(), h_a.data(), params.m * params.n * params.batch_size,
               handle.get_stream());
    raft::copy(x.data(), h_x.data(), params.n * params.batch_size,
               handle.get_stream());

    /* Compute reference results */
    for (int bid = 0; bid < params.batch_size; bid++) {
      for (int i = 0; i < params.m; i++) {
        T acc = (T)0;
        for (int j = 0; j < params.n; j++) {
          acc += h_a[bid * params.m * params.n + j * params.m + i] *
                 h_x[bid * params.n + j];
        }
        h_y_ref[bid * params.m + i] = alpha * acc;
      }
    }

    /* Compute using tested prims */
    using Policy = BlockGemvPolicy<32, 8>;
    int shared_mem_size = params.n * sizeof(T);
    block_gemv_test_kernel<Policy><<<params.batch_size, Policy::BlockSize,
                                     shared_mem_size, handle.get_stream()>>>(
      params.m, params.n, alpha, a.data(), x.data(), y.data());

    /* Copy results to host */
    raft::copy(h_y_dev.data(), y.data(), params.m * params.batch_size,
               handle.get_stream());
    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    /* Count errors */
    errors = 0;
    for (int i = 0; i < params.m * params.batch_size; i++) {
      T diff = abs(h_y_ref[i] - h_y_dev[i]);
      if (std::isnan(diff) || diff > params.eps) errors++;
    }
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  BlockGemvInputs<T> params;

  int errors;
};

const std::vector<BlockGemvInputs<float>> gemv_inputsf = {
  {42, 42, 20, 1e-4, 12345U},
  {65, 10, 50, 1e-4, 12345U},
  {5, 80, 100, 1e-4, 12345U}};

const std::vector<BlockGemvInputs<double>> gemv_inputsd = {
  {42, 42, 20, 1e-4, 12345U},
  {65, 10, 50, 1e-4, 12345U},
  {5, 80, 100, 1e-4, 12345U}};

typedef BlockGemvTest<float> BlockGemvTestF;
TEST_P(BlockGemvTestF, Result) { ASSERT_EQ(errors, 0); }

typedef BlockGemvTest<double> BlockGemvTestD;
TEST_P(BlockGemvTestD, Result) { ASSERT_EQ(errors, 0); }

INSTANTIATE_TEST_CASE_P(BlockGemvTests, BlockGemvTestF,
                        ::testing::ValuesIn(gemv_inputsf));

INSTANTIATE_TEST_CASE_P(BlockGemvTests, BlockGemvTestD,
                        ::testing::ValuesIn(gemv_inputsd));

/* DOT */

template <typename T>
struct BlockDotInputs {
  bool broadcast;
  int n;
  int batch_size;
  T eps;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const BlockDotInputs<T>& dims) {
  return os;
}

template <int BlockSize, bool Broadcast, typename T>
__global__ void block_dot_test_kernel(int n, const T* x, const T* y, T* d_dot) {
  __shared__ ReductionStorage<BlockSize, T> reduction_storage;

  T dot_ = _block_dot<BlockSize, Broadcast>(
    n, x + n * blockIdx.x, y + n * blockIdx.x, reduction_storage);

  if (!Broadcast && threadIdx.x == 0)
    d_dot[blockIdx.x] = dot_;
  else if (Broadcast && threadIdx.x == BlockSize - 1)
    d_dot[blockIdx.x] = dot_;
}

template <typename T>
class BlockDotTest : public ::testing::TestWithParam<BlockDotInputs<T>> {
 protected:
  void basicTest() {
    raft::handle_t handle;

    params = ::testing::TestWithParam<BlockDotInputs<T>>::GetParam();

    device_buffer<T> x(handle.get_device_allocator(), handle.get_stream(),
                       params.n * params.batch_size);
    device_buffer<T> y(handle.get_device_allocator(), handle.get_stream(),
                       params.n * params.batch_size);
    device_buffer<T> dot_dev(handle.get_device_allocator(), handle.get_stream(),
                             params.batch_size);

    std::vector<T> h_x(params.n * params.batch_size);
    std::vector<T> h_y(params.n * params.batch_size);
    std::vector<T> h_dot_ref(params.batch_size, (T)0);
    std::vector<T> h_dot_dev(params.batch_size);

    /* Generate random data */
    std::default_random_engine generator(params.seed);
    std::uniform_real_distribution<double> distribution(-2.0, 2.0);
    for (int i = 0; i < params.n * params.batch_size; i++)
      h_x[i] = distribution(generator);
    for (int i = 0; i < params.n * params.batch_size; i++)
      h_y[i] = distribution(generator);

    /* Copy to device */
    raft::copy(x.data(), h_x.data(), params.n * params.batch_size,
               handle.get_stream());
    raft::copy(y.data(), h_y.data(), params.n * params.batch_size,
               handle.get_stream());

    /* Compute reference results */
    for (int bid = 0; bid < params.batch_size; bid++) {
      for (int i = 0; i < params.n; i++) {
        h_dot_ref[bid] += h_x[bid * params.n + i] * h_y[bid * params.n + i];
      }
    }

    /* Compute using tested prims */
    constexpr int BlockSize = 64;
    if (params.broadcast)
      block_dot_test_kernel<BlockSize, true>
        <<<params.batch_size, BlockSize, 0, handle.get_stream()>>>(
          params.n, x.data(), y.data(), dot_dev.data());
    else
      block_dot_test_kernel<BlockSize, false>
        <<<params.batch_size, BlockSize, 0, handle.get_stream()>>>(
          params.n, x.data(), y.data(), dot_dev.data());

    /* Copy results to host */
    raft::copy(h_dot_dev.data(), dot_dev.data(), params.batch_size,
               handle.get_stream());
    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    /* Count errors */
    errors = 0;
    for (int i = 0; i < params.batch_size; i++) {
      T diff = abs(h_dot_dev[i] - h_dot_ref[i]);
      if (std::isnan(diff) || diff > params.eps) errors++;
    }
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  BlockDotInputs<T> params;

  int errors;
};

const std::vector<BlockDotInputs<float>> dot_inputsf = {
  {true, 9, 20, 1e-4, 12345U},
  {true, 65, 50, 1e-4, 12345U},
  {true, 200, 100, 1e-4, 12345U},
  {false, 200, 100, 1e-4, 12345U}};

const std::vector<BlockDotInputs<double>> dot_inputsd = {
  {true, 9, 20, 1e-4, 12345U},
  {true, 65, 50, 1e-4, 12345U},
  {true, 200, 100, 1e-4, 12345U},
  {false, 200, 100, 1e-4, 12345U}};

typedef BlockDotTest<float> BlockDotTestF;
TEST_P(BlockDotTestF, Result) { ASSERT_EQ(errors, 0); }

typedef BlockDotTest<double> BlockDotTestD;
TEST_P(BlockDotTestD, Result) { ASSERT_EQ(errors, 0); }

INSTANTIATE_TEST_CASE_P(BlockDotTests, BlockDotTestF,
                        ::testing::ValuesIn(dot_inputsf));

INSTANTIATE_TEST_CASE_P(BlockDotTests, BlockDotTestD,
                        ::testing::ValuesIn(dot_inputsd));

/* xAx' */

template <typename T>
struct BlockXaxtInputs {
  bool broadcast;
  int n;
  int batch_size;
  T eps;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const BlockXaxtInputs<T>& dims) {
  return os;
}

template <int BlockSize, bool Broadcast, typename T>
__global__ void block_xAxt_test_kernel(int n, const T* x, const T* A,
                                       T* d_res) {
  extern __shared__ char dyna_shared_mem[];
  T* shared_vec = (T*)dyna_shared_mem;
  __shared__ ReductionStorage<BlockSize, T> reduction_storage;

  T res_ = _block_xAxt<BlockSize, Broadcast>(n, x + n * blockIdx.x,
                                             A + n * n * blockIdx.x,
                                             reduction_storage, shared_vec);

  if (!Broadcast && threadIdx.x == 0)
    d_res[blockIdx.x] = res_;
  else if (Broadcast && threadIdx.x == BlockSize - 1)
    d_res[blockIdx.x] = res_;
}

template <typename T>
class BlockXaxtTest : public ::testing::TestWithParam<BlockXaxtInputs<T>> {
 protected:
  void basicTest() {
    raft::handle_t handle;

    params = ::testing::TestWithParam<BlockXaxtInputs<T>>::GetParam();

    device_buffer<T> x(handle.get_device_allocator(), handle.get_stream(),
                       params.n * params.batch_size);
    device_buffer<T> A(handle.get_device_allocator(), handle.get_stream(),
                       params.n * params.n * params.batch_size);
    device_buffer<T> res_dev(handle.get_device_allocator(), handle.get_stream(),
                             params.batch_size);

    std::vector<T> h_x(params.n * params.batch_size);
    std::vector<T> h_A(params.n * params.n * params.batch_size);
    std::vector<T> h_res_ref(params.batch_size, (T)0);
    std::vector<T> h_res_dev(params.batch_size);

    /* Generate random data */
    std::default_random_engine generator(params.seed);
    std::uniform_real_distribution<double> distribution(-2.0, 2.0);
    for (int i = 0; i < params.n * params.batch_size; i++)
      h_x[i] = distribution(generator);
    for (int i = 0; i < params.n * params.n * params.batch_size; i++)
      h_A[i] = distribution(generator);

    /* Copy to device */
    raft::copy(x.data(), h_x.data(), params.n * params.batch_size,
               handle.get_stream());
    raft::copy(A.data(), h_A.data(), params.n * params.n * params.batch_size,
               handle.get_stream());

    /* Compute reference results */
    for (int bid = 0; bid < params.batch_size; bid++) {
      for (int i = 0; i < params.n; i++) {
        T acc = 0;
        for (int j = 0; j < params.n; j++) {
          acc += h_A[bid * params.n * params.n + j * params.n + i] *
                 h_x[bid * params.n + j];
        }
        h_res_ref[bid] += acc * h_x[bid * params.n + i];
      }
    }

    /* Compute using tested prims */
    constexpr int BlockSize = 64;
    int shared_mem_size = params.n * sizeof(T);
    if (params.broadcast)
      block_xAxt_test_kernel<BlockSize, true>
        <<<params.batch_size, BlockSize, shared_mem_size,
           handle.get_stream()>>>(params.n, x.data(), A.data(), res_dev.data());
    else
      block_xAxt_test_kernel<BlockSize, false>
        <<<params.batch_size, BlockSize, shared_mem_size,
           handle.get_stream()>>>(params.n, x.data(), A.data(), res_dev.data());

    /* Copy results to host */
    raft::copy(h_res_dev.data(), res_dev.data(), params.batch_size,
               handle.get_stream());
    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    /* Count errors */
    errors = 0;
    for (int i = 0; i < params.batch_size; i++) {
      T diff = abs(h_res_dev[i] - h_res_ref[i]);
      if (std::isnan(diff) || diff > params.eps) errors++;
    }
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  BlockXaxtInputs<T> params;

  int errors;
};

const std::vector<BlockXaxtInputs<float>> xAxt_inputsf = {
  {true, 9, 20, 1e-2, 12345U},
  {true, 65, 50, 1e-2, 12345U},
  {true, 200, 100, 1e-2, 12345U},
  {false, 200, 100, 1e-2, 12345U}};

const std::vector<BlockXaxtInputs<double>> xAxt_inputsd = {
  {true, 9, 20, 1e-4, 12345U},
  {true, 65, 50, 1e-4, 12345U},
  {true, 200, 100, 1e-4, 12345U},
  {false, 200, 100, 1e-4, 12345U}};

typedef BlockXaxtTest<float> BlockXaxtTestF;
TEST_P(BlockXaxtTestF, Result) { ASSERT_EQ(errors, 0); }

typedef BlockXaxtTest<double> BlockXaxtTestD;
TEST_P(BlockXaxtTestD, Result) { ASSERT_EQ(errors, 0); }

INSTANTIATE_TEST_CASE_P(BlockXaxtTests, BlockXaxtTestF,
                        ::testing::ValuesIn(xAxt_inputsf));

INSTANTIATE_TEST_CASE_P(BlockXaxtTests, BlockXaxtTestD,
                        ::testing::ValuesIn(xAxt_inputsd));

}  // namespace LinAlg
}  // namespace MLCommon
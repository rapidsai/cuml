/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <raft/random/rng.cuh>

#include <test_utils.h>

#include <cuml/common/device_buffer.hpp>
#include <cuml/common/logger.hpp>

#include <linalg/block.cuh>

namespace MLCommon {
namespace LinAlg {

using namespace std;

/// TODO: test both preloaded and non-preloaded to shared memory

/* GEMM */

template <typename T>
struct BlockGemmInputs {
  int m, k, n;
  bool transa, transb;
  int batch_size;
  int vec_len;
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

    /* Generate random data on device */
    raft::random::Rng r(params.seed);
    r.uniform(a.data(), params.m * params.k * params.batch_size, (T)-2, (T)2,
              handle.get_stream());
    r.uniform(b.data(), params.k * params.n * params.batch_size, (T)-2, (T)2,
              handle.get_stream());

    /* Generate random alpha */
    std::default_random_engine generator(params.seed);
    std::uniform_real_distribution<T> distribution(-2.0, 2.0);
    T alpha = distribution(generator);

    /* Copy to host */
    raft::update_host(h_a.data(), a.data(),
                      params.m * params.k * params.batch_size,
                      handle.get_stream());
    raft::update_host(h_b.data(), b.data(),
                      params.k * params.n * params.batch_size,
                      handle.get_stream());
    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

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
    if (params.vec_len == 1) {
      using Policy = BlockGemmPolicy<1, 32, 2, 2, 16, 16>;
      block_gemm_test_kernel<Policy>
        <<<params.batch_size, Policy::BlockSize, 0, handle.get_stream()>>>(
          params.transa, params.transb, params.m, params.n, params.k, alpha,
          a.data(), b.data(), c.data());
    } else if (params.vec_len == 2) {
      using Policy = BlockGemmPolicy<2, 32, 2, 2, 16, 16>;
      block_gemm_test_kernel<Policy>
        <<<params.batch_size, Policy::BlockSize, 0, handle.get_stream()>>>(
          params.transa, params.transb, params.m, params.n, params.k, alpha,
          a.data(), b.data(), c.data());
    }

    /* Check results */
    match = devArrMatchHost(
      h_c_ref.data(), c.data(), params.m * params.n * params.batch_size,
      raft::CompareApprox<T>(params.eps), handle.get_stream());
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  BlockGemmInputs<T> params;

  testing::AssertionResult match = testing::AssertionFailure();
};

const std::vector<BlockGemmInputs<float>> gemm_inputsf = {
  {42, 42, 42, false, false, 20, 1, 1e-4, 12345U},
  {65, 10, 20, false, true, 50, 1, 1e-4, 12345U},
  {5, 80, 31, true, false, 80, 1, 1e-4, 12345U},
  {11, 50, 41, true, true, 100, 1, 1e-4, 12345U},
  {30, 34, 16, false, false, 20, 2, 1e-4, 12345U},
  {10, 42, 20, false, true, 20, 2, 1e-4, 12345U},
  {14, 8, 22, true, false, 20, 2, 1e-4, 12345U},
  {56, 72, 28, true, true, 20, 2, 1e-4, 12345U},
};

const std::vector<BlockGemmInputs<double>> gemm_inputsd = {
  {42, 42, 42, false, false, 20, 1, 1e-4, 12345U},
  {65, 10, 20, false, true, 50, 1, 1e-4, 12345U},
  {5, 80, 31, true, false, 80, 1, 1e-4, 12345U},
  {11, 50, 41, true, true, 100, 1, 1e-4, 12345U},
  {30, 34, 16, false, false, 20, 2, 1e-4, 12345U},
  {10, 42, 20, false, true, 20, 2, 1e-4, 12345U},
  {14, 8, 22, true, false, 20, 2, 1e-4, 12345U},
  {56, 72, 28, true, true, 20, 2, 1e-4, 12345U},
};

typedef BlockGemmTest<float> BlockGemmTestF;
TEST_P(BlockGemmTestF, Result) { EXPECT_TRUE(match); }

typedef BlockGemmTest<double> BlockGemmTestD;
TEST_P(BlockGemmTestD, Result) { EXPECT_TRUE(match); }

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

  _block_gemv<Policy, true>(m, n, alpha, a + m * n * blockIdx.x,
                            x + n * blockIdx.x, y + m * blockIdx.x,
                            gemv_storage, shared_vec);
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

    /* Generate random data on device */
    raft::random::Rng r(params.seed);
    r.uniform(a.data(), params.m * params.n * params.batch_size, (T)-2, (T)2,
              handle.get_stream());
    r.uniform(x.data(), params.n * params.batch_size, (T)-2, (T)2,
              handle.get_stream());

    /* Generate random alpha */
    std::default_random_engine generator(params.seed);
    std::uniform_real_distribution<T> distribution(-2.0, 2.0);
    T alpha = distribution(generator);

    /* Copy to host */
    raft::update_host(h_a.data(), a.data(),
                      params.m * params.n * params.batch_size,
                      handle.get_stream());
    raft::update_host(h_x.data(), x.data(), params.n * params.batch_size,
                      handle.get_stream());
    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

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

    /* Check results */
    match =
      devArrMatchHost(h_y_ref.data(), y.data(), params.m * params.batch_size,
                      raft::CompareApprox<T>(params.eps), handle.get_stream());
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  BlockGemvInputs<T> params;

  testing::AssertionResult match = testing::AssertionFailure();
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
TEST_P(BlockGemvTestF, Result) { EXPECT_TRUE(match); }

typedef BlockGemvTest<double> BlockGemvTestD;
TEST_P(BlockGemvTestD, Result) { EXPECT_TRUE(match); }

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

    /* Generate random data on device */
    raft::random::Rng r(params.seed);
    r.uniform(x.data(), params.n * params.batch_size, (T)-2, (T)2,
              handle.get_stream());
    r.uniform(y.data(), params.n * params.batch_size, (T)-2, (T)2,
              handle.get_stream());

    /* Copy to host */
    raft::update_host(h_x.data(), x.data(), params.n * params.batch_size,
                      handle.get_stream());
    raft::update_host(h_y.data(), y.data(), params.n * params.batch_size,
                      handle.get_stream());
    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

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

    /* Check results */
    match =
      devArrMatchHost(h_dot_ref.data(), dot_dev.data(), params.batch_size,
                      raft::CompareApprox<T>(params.eps), handle.get_stream());
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  BlockDotInputs<T> params;

  testing::AssertionResult match = testing::AssertionFailure();
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
TEST_P(BlockDotTestF, Result) { EXPECT_TRUE(match); }

typedef BlockDotTest<double> BlockDotTestD;
TEST_P(BlockDotTestD, Result) { EXPECT_TRUE(match); }

INSTANTIATE_TEST_CASE_P(BlockDotTests, BlockDotTestF,
                        ::testing::ValuesIn(dot_inputsf));

INSTANTIATE_TEST_CASE_P(BlockDotTests, BlockDotTestD,
                        ::testing::ValuesIn(dot_inputsd));

/* x*A*x' */

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

  T res_ = _block_xAxt<BlockSize, Broadcast, true>(
    n, x + n * blockIdx.x, A + n * n * blockIdx.x, reduction_storage,
    shared_vec);

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

    /* Generate random data on device */
    raft::random::Rng r(params.seed);
    r.uniform(x.data(), params.n * params.batch_size, (T)-2, (T)2,
              handle.get_stream());
    r.uniform(A.data(), params.n * params.n * params.batch_size, (T)-2, (T)2,
              handle.get_stream());

    /* Copy to host */
    raft::update_host(h_x.data(), x.data(), params.n * params.batch_size,
                      handle.get_stream());
    raft::update_host(h_A.data(), A.data(),
                      params.n * params.n * params.batch_size,
                      handle.get_stream());
    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

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

    /* Check results */
    match =
      devArrMatchHost(h_res_ref.data(), res_dev.data(), params.batch_size,
                      raft::CompareApprox<T>(params.eps), handle.get_stream());
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  BlockXaxtInputs<T> params;

  testing::AssertionResult match = testing::AssertionFailure();
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
TEST_P(BlockXaxtTestF, Result) { EXPECT_TRUE(match); }

typedef BlockXaxtTest<double> BlockXaxtTestD;
TEST_P(BlockXaxtTestD, Result) { EXPECT_TRUE(match); }

INSTANTIATE_TEST_CASE_P(BlockXaxtTests, BlockXaxtTestF,
                        ::testing::ValuesIn(xAxt_inputsf));

INSTANTIATE_TEST_CASE_P(BlockXaxtTests, BlockXaxtTestD,
                        ::testing::ValuesIn(xAxt_inputsd));

/* y=alpha*x */

template <typename T>
struct BlockAxInputs {
  int n;
  int batch_size;
  T eps;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const BlockAxInputs<T>& dims) {
  return os;
}

template <typename T>
__global__ void block_ax_test_kernel(int n, T alpha, const T* x, T* y) {
  _block_ax(n, alpha, x + n * blockIdx.x, y + n * blockIdx.x);
}

template <typename T>
class BlockAxTest : public ::testing::TestWithParam<BlockAxInputs<T>> {
 protected:
  void basicTest() {
    raft::handle_t handle;

    params = ::testing::TestWithParam<BlockAxInputs<T>>::GetParam();

    device_buffer<T> x(handle.get_device_allocator(), handle.get_stream(),
                       params.n * params.batch_size);
    device_buffer<T> y(handle.get_device_allocator(), handle.get_stream(),
                       params.n * params.batch_size);

    std::vector<T> h_x(params.n * params.batch_size);
    std::vector<T> h_y_ref(params.n * params.batch_size, (T)0);

    /* Generate random data on device */
    raft::random::Rng r(params.seed);
    r.uniform(x.data(), params.n * params.batch_size, (T)-2, (T)2,
              handle.get_stream());

    /* Generate random alpha */
    std::default_random_engine generator(params.seed);
    std::uniform_real_distribution<T> distribution(-2.0, 2.0);
    T alpha = distribution(generator);

    /* Copy to host */
    raft::update_host(h_x.data(), x.data(), params.n * params.batch_size,
                      handle.get_stream());
    CUDA_CHECK(cudaStreamSynchronize(handle.get_stream()));

    /* Compute reference results */
    for (int bid = 0; bid < params.batch_size; bid++) {
      for (int i = 0; i < params.n; i++) {
        h_y_ref[bid * params.n + i] = alpha * h_x[bid * params.n + i];
      }
    }

    /* Compute using tested prims */
    constexpr int BlockSize = 64;
    block_ax_test_kernel<<<params.batch_size, BlockSize, 0,
                           handle.get_stream()>>>(params.n, alpha, x.data(),
                                                  y.data());

    /* Check results */
    match =
      devArrMatchHost(h_y_ref.data(), y.data(), params.n * params.batch_size,
                      raft::CompareApprox<T>(params.eps), handle.get_stream());
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  BlockAxInputs<T> params;

  testing::AssertionResult match = testing::AssertionFailure();
};

const std::vector<BlockAxInputs<float>> ax_inputsf = {
  {9, 20, 1e-4, 12345U}, {65, 50, 1e-4, 12345U}, {200, 100, 1e-4, 12345U}};

const std::vector<BlockAxInputs<double>> ax_inputsd = {
  {9, 20, 1e-4, 12345U}, {65, 50, 1e-4, 12345U}, {200, 100, 1e-4, 12345U}};

typedef BlockAxTest<float> BlockAxTestF;
TEST_P(BlockAxTestF, Result) { EXPECT_TRUE(match); }

typedef BlockAxTest<double> BlockAxTestD;
TEST_P(BlockAxTestD, Result) { EXPECT_TRUE(match); }

INSTANTIATE_TEST_CASE_P(BlockAxTests, BlockAxTestF,
                        ::testing::ValuesIn(ax_inputsf));

INSTANTIATE_TEST_CASE_P(BlockAxTests, BlockAxTestD,
                        ::testing::ValuesIn(ax_inputsd));

}  // namespace LinAlg
}  // namespace MLCommon
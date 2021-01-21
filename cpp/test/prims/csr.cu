/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <sparse/csr.cuh>
#include "csr.h"

#include <raft/cudart_utils.h>
#include <raft/random/rng.cuh>
#include "test_utils.h"

#include <iostream>
#include <limits>

namespace raft {
namespace sparse {

template <typename T>
class CSRTest : public ::testing::TestWithParam<CSRInputs<T>> {
 protected:
  void SetUp() override {}

  void TearDown() override {}

 protected:
  CSRInputs<T> params;
};

const std::vector<CSRInputs<float>> inputsf = {{5, 10, 5, 1234ULL}};

typedef CSRTest<float> WeakCCTest;
TEST_P(WeakCCTest, Result) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::shared_ptr<MLCommon::deviceAllocator> alloc(
    new raft::mr::device::default_allocator);
  int *row_ind, *row_ind_ptr, *result, *verify;

  int row_ind_h1[3] = {0, 3, 6};
  int row_ind_ptr_h1[9] = {0, 1, 2, 0, 1, 2, 0, 1, 2};
  int verify_h1[6] = {1, 1, 1, 2147483647, 2147483647, 2147483647};

  int row_ind_h2[3] = {0, 2, 4};
  int row_ind_ptr_h2[5] = {3, 4, 3, 4, 5};
  int verify_h2[6] = {1, 1, 1, 5, 5, 5};

  raft::allocate(row_ind, 3);
  raft::allocate(row_ind_ptr, 9);
  raft::allocate(result, 9, true);
  raft::allocate(verify, 9);

  MLCommon::device_buffer<bool> xa(alloc, stream, 6);
  MLCommon::device_buffer<bool> fa(alloc, stream, 6);
  MLCommon::device_buffer<bool> m(alloc, stream, 1);
  WeakCCState state(xa.data(), fa.data(), m.data());

  /**
     * Run batch #1
     */
  raft::update_device(row_ind, *&row_ind_h1, 3, stream);
  raft::update_device(row_ind_ptr, *&row_ind_ptr_h1, 9, stream);
  raft::update_device(verify, *&verify_h1, 6, stream);

  weak_cc_batched<int, 32>(result, row_ind, row_ind_ptr, 9, 6, 0, 3, &state,
                           stream);

  cudaStreamSynchronize(stream);
  ASSERT_TRUE(raft::devArrMatch<int>(verify, result, 6, raft::Compare<int>()));

  /**
     * Run batch #2
     */
  raft::update_device(row_ind, *&row_ind_h2, 3, stream);
  raft::update_device(row_ind_ptr, *&row_ind_ptr_h2, 5, stream);
  raft::update_device(verify, *&verify_h2, 6, stream);

  weak_cc_batched<int, 32>(result, row_ind, row_ind_ptr, 5, 6, 4, 3, &state,
                           stream);

  ASSERT_TRUE(raft::devArrMatch<int>(verify, result, 6, raft::Compare<int>()));

  cudaStreamSynchronize(stream);

  cudaStreamDestroy(stream);

  CUDA_CHECK(cudaFree(row_ind));
  CUDA_CHECK(cudaFree(row_ind_ptr));
  CUDA_CHECK(cudaFree(verify));
  CUDA_CHECK(cudaFree(result));
}

INSTANTIATE_TEST_CASE_P(CSRTests, WeakCCTest, ::testing::ValuesIn(inputsf));

}  // namespace sparse
}  // namespace raft

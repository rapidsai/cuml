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
#include <raft/cudart_utils.h>
#include <raft/random/rng.cuh>
#include <test_utils.h>

#include <sparse/coo.cuh>
#include <sparse/linalg/symmetrize.cuh>

#include <iostream>

namespace raft {
namespace sparse {

template <typename T>
struct SparseSymmetrizeInput {
  int m, n, nnz;
  unsigned long long int seed;
};

template <typename T>
class SparseSymmetrizeTest
  : public ::testing::TestWithParam<SparseSymmetrizeInput<T>> {
 protected:
  void SetUp() override {}

  void TearDown() override {}

 protected:
  SparseSymmetrizeInput<T> params;
};

const std::vector<SparseSymmetrizeInput<float>> inputsf = {{5, 10, 5, 1234ULL}};

typedef SparseSymmetrizeTest<float> COOSymmetrize;
TEST_P(COOSymmetrize, Result) {
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::shared_ptr<MLCommon::deviceAllocator> alloc(
    new raft::mr::device::default_allocator);

  int nnz = 8;

  int *in_rows_h = new int[nnz]{0, 0, 1, 1, 2, 2, 3, 3};
  int *in_cols_h = new int[nnz]{1, 3, 2, 3, 0, 1, 0, 2};
  float *in_vals_h = new float[nnz]{0.5, 1.0, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5};

  int *exp_rows_h =
    new int[nnz * 2]{1, 0, 0, 0, 1, 3, 1, 0, 0, 2, 2, 0, 3, 2, 3, 0};
  int *exp_cols_h =
    new int[nnz * 2]{0, 1, 3, 0, 2, 1, 3, 0, 2, 0, 1, 0, 0, 3, 2, 0};
  float *exp_vals_h = new float[nnz * 2]{0.5, 0.5, 1.5, 0, 0.5, 0.5, 0.5, 0,
                                         0.5, 0.5, 0.5, 0, 1.5, 0.5, 0.5, 0.0};

  COO<float> in(alloc, stream, nnz, 4, 4);
  raft::update_device(in.rows(), *&in_rows_h, nnz, stream);
  raft::update_device(in.cols(), *&in_cols_h, nnz, stream);
  raft::update_device(in.vals(), *&in_vals_h, nnz, stream);

  COO<float> out(alloc, stream);

  linalg::coo_symmetrize<32, float>(
    &in, &out,
    [] __device__(int row, int col, float val, float trans) {
      return val + trans;
    },
    alloc, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream));
  std::cout << out << std::endl;

  ASSERT_TRUE(out.nnz == nnz * 2);
  ASSERT_TRUE(raft::devArrMatch<int>(out.rows(), exp_rows_h, out.nnz,
                                     raft::Compare<int>()));
  ASSERT_TRUE(raft::devArrMatch<int>(out.cols(), exp_cols_h, out.nnz,
                                     raft::Compare<int>()));
  ASSERT_TRUE(raft::devArrMatch<float>(out.vals(), exp_vals_h, out.nnz,
                                       raft::Compare<float>()));

  cudaStreamDestroy(stream);

  delete[] in_rows_h;
  delete[] in_cols_h;
  delete[] in_vals_h;

  delete[] exp_rows_h;
  delete[] exp_cols_h;
  delete[] exp_vals_h;
}

INSTANTIATE_TEST_CASE_P(SparseSymmetrizeTest, COOSymmetrize,
                        ::testing::ValuesIn(inputsf));

}  // namespace sparse
}  // namespace raft

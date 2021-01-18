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

#include <sparse/convert/coo.cuh>
#include <sparse/csr.cuh>

#include <raft/cudart_utils.h>
#include <raft/random/rng.cuh>

#include <test_utils.h>

#include <iostream>
#include <limits>

namespace raft {
namespace sparse {

template <typename Index_>
struct CSRtoCOOInputs {
  std::vector<Index_> ex_scan;
  std::vector<Index_> verify;
};

template <typename Index_>
class CSRtoCOOTest : public ::testing::TestWithParam<CSRtoCOOInputs<Index_>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<CSRtoCOOInputs<Index_>>::GetParam();

    cudaStreamCreate(&stream);
    raft::allocate(ex_scan, params.ex_scan.size());
    raft::allocate(verify, params.verify.size());
    raft::allocate(result, params.verify.size(), true);
  }

  void Run() {
    Index_ n_rows = params.ex_scan.size();
    Index_ nnz = params.verify.size();

    raft::update_device(ex_scan, params.ex_scan.data(), n_rows, stream);
    raft::update_device(verify, params.verify.data(), nnz, stream);

    convert::csr_to_coo<Index_, 32>(ex_scan, n_rows, result, nnz, stream);

    ASSERT_TRUE(raft::devArrMatch<Index_>(verify, result, nnz,
                                          raft::Compare<float>(), stream));
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(ex_scan));
    CUDA_CHECK(cudaFree(verify));
    CUDA_CHECK(cudaFree(result));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

 protected:
  CSRtoCOOInputs<Index_> params;
  cudaStream_t stream;
  Index_ *ex_scan, *verify, *result;
};

using CSRtoCOOTestI = CSRtoCOOTest<int>;
TEST_P(CSRtoCOOTestI, Result) { Run(); }

using CSRtoCOOTestL = CSRtoCOOTest<int64_t>;
TEST_P(CSRtoCOOTestL, Result) { Run(); }

const std::vector<CSRtoCOOInputs<int>> csrtocoo_inputs_32 = {
  {{0, 0, 2, 2}, {1, 1, 3}},
  {{0, 4, 8, 9}, {0, 0, 0, 0, 1, 1, 1, 1, 2, 3}},
};
const std::vector<CSRtoCOOInputs<int64_t>> csrtocoo_inputs_64 = {
  {{0, 0, 2, 2}, {1, 1, 3}},
  {{0, 4, 8, 9}, {0, 0, 0, 0, 1, 1, 1, 1, 2, 3}},
};

INSTANTIATE_TEST_CASE_P(SparseConvertCOOTest, CSRtoCOOTestI,
                        ::testing::ValuesIn(csrtocoo_inputs_32));
INSTANTIATE_TEST_CASE_P(SparseConvertCOOTest, CSRtoCOOTestL,
                        ::testing::ValuesIn(csrtocoo_inputs_64));

}  // namespace sparse
}  // namespace raft

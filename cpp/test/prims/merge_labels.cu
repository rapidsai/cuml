/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <label/merge_labels.cuh>

#include <raft/cudart_utils.h>
#include <cuml/common/cuml_allocator.hpp>
#include "test_utils.h"

// #include <iostream>
// #include <limits>
#include <vector>

namespace MLCommon {
namespace Label {

template <typename Index_>
struct MergeLabelsInputs {
  Index_ N;
  std::vector<Index_> labelsA;
  std::vector<Index_> labelsB;
  std::vector<uint8_t> mask;  // to avoid std::vector<bool> optimization
  std::vector<Index_> expected;
};

template <typename Index_>
class MergeLabelsTest
  : public ::testing::TestWithParam<MergeLabelsInputs<Index_>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<MergeLabelsInputs<Index_>>::GetParam();

    CUDA_CHECK(cudaStreamCreate(&stream));
    std::shared_ptr<deviceAllocator> alloc(
      new raft::mr::device::default_allocator);

    raft::allocate(labelsA, params.N);
    raft::allocate(labelsB, params.N);
    raft::allocate(expected, params.N);
    raft::allocate(R, params.N);
    raft::allocate(mask, params.N);
    raft::allocate(m, 1);
  }

  void Run() {
    params = ::testing::TestWithParam<MergeLabelsInputs<Index_>>::GetParam();

    raft::update_device(labelsA, params.labelsA.data(), params.N, stream);
    raft::update_device(labelsB, params.labelsB.data(), params.N, stream);
    raft::update_device(expected, params.expected.data(), params.N, stream);
    raft::update_device(mask, reinterpret_cast<bool *>(params.mask.data()),
                        params.N, stream);

    merge_labels(labelsA, labelsB, mask, R, m, params.N, stream);

    cudaStreamSynchronize(stream);
    ASSERT_TRUE(raft::devArrMatch<Index_>(expected, labelsA, params.N,
                                          raft::Compare<Index_>()));
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(labelsA));
    CUDA_CHECK(cudaFree(labelsB));
    CUDA_CHECK(cudaFree(expected));
    CUDA_CHECK(cudaFree(R));
    CUDA_CHECK(cudaFree(mask));
    CUDA_CHECK(cudaFree(m));
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

 protected:
  MergeLabelsInputs<Index_> params;
  cudaStream_t stream;
  Index_ *labelsA, *labelsB, *expected, *R;
  bool *mask, *m;
};

using MergeLabelsTestI = MergeLabelsTest<int>;
TEST_P(MergeLabelsTestI, Result) { Run(); }

using MergeLabelsTestL = MergeLabelsTest<int64_t>;
TEST_P(MergeLabelsTestL, Result) { Run(); }

constexpr int MAX32 = std::numeric_limits<int>::max();
constexpr int64_t MAX64 = std::numeric_limits<int64_t>::max();

const std::vector<MergeLabelsInputs<int>> merge_inputs_32 = {
  {4, {1, 1, 3, MAX32}, {1, 3, 3, 1}, {1, 0, 1, 0}, {1, 1, 3, 1}},
  {5, {1, 2, 2, 2, 1}, {4, 2, 4, 4, 4}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
  {6,
   {1, 2, 1, 4, 5, MAX32},
   {1, 2, MAX32, 4, 5, 4},
   {1, 1, 0, 1, 1, 0},
   {1, 2, 1, 4, 5, 4}},
  {6,
   {1, 2, 2, 2, 2, 6},
   {1, 1, 1, 5, 5, 5},
   {1, 1, 1, 1, 1, 1},
   {1, 1, 1, 1, 1, 1}},
  {8,
   {1, 1, 3, 3, MAX32, 1, 3, MAX32},
   {1, 2, 3, 2, MAX32, 2, 2, 2},
   {1, 1, 1, 1, 0, 1, 1, 0},
   {1, 1, 1, 1, MAX32, 1, 1, 1}},
  {8,
   {1, 1, 1, 4, 4, 7, 7, 8},
   {1, 2, 2, 2, 2, 7, 7, 7},
   {1, 1, 1, 1, 0, 0, 1, 1},
   {1, 1, 1, 1, 1, 7, 7, 7}},
};

const std::vector<MergeLabelsInputs<int64_t>> merge_inputs_64 = {
  {4, {1, 1, 3, MAX64}, {1, 3, 3, 1}, {1, 0, 1, 0}, {1, 1, 3, 1}},
  {5, {1, 2, 2, 2, 1}, {4, 2, 4, 4, 4}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
  {6,
   {1, 2, 1, 4, 5, MAX64},
   {1, 2, MAX64, 4, 5, 4},
   {1, 1, 0, 1, 1, 0},
   {1, 2, 1, 4, 5, 4}},
  {6,
   {1, 2, 2, 2, 2, 6},
   {1, 1, 1, 5, 5, 5},
   {1, 1, 1, 1, 1, 1},
   {1, 1, 1, 1, 1, 1}},
  {8,
   {1, 1, 3, 3, MAX64, 1, 3, MAX64},
   {1, 2, 3, 2, MAX64, 2, 2, 2},
   {1, 1, 1, 1, 0, 1, 1, 0},
   {1, 1, 1, 1, MAX64, 1, 1, 1}},
  {8,
   {1, 1, 1, 4, 4, 7, 7, 8},
   {1, 2, 2, 2, 2, 7, 7, 7},
   {1, 1, 1, 1, 0, 0, 1, 1},
   {1, 1, 1, 1, 1, 7, 7, 7}},
};

INSTANTIATE_TEST_CASE_P(MergeLabelsTests, MergeLabelsTestI,
                        ::testing::ValuesIn(merge_inputs_32));
INSTANTIATE_TEST_CASE_P(MergeLabelsTests, MergeLabelsTestL,
                        ::testing::ValuesIn(merge_inputs_64));

}  // namespace Label
}  // namespace MLCommon

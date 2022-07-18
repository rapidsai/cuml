/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include "test_utils.h"
#include <raft/cudart_utils.h>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <thrust/device_ptr.h>

#include <vector>

namespace MLCommon {
namespace Label {

template <typename Index_>
struct MergeLabelsInputs {
  Index_ N;
  std::vector<Index_> labels_a;
  std::vector<Index_> labels_b;
  std::vector<uint8_t> mask;  // to avoid std::vector<bool> optimization
  std::vector<Index_> expected;
};

template <typename Index_>
class MergeLabelsTest : public ::testing::TestWithParam<MergeLabelsInputs<Index_>> {
 protected:
  MergeLabelsTest()
    : params(::testing::TestWithParam<MergeLabelsInputs<Index_>>::GetParam()),
      stream(handle.get_stream()),
      labels_a(params.N, stream),
      labels_b(params.N, stream),
      expected(params.N, stream),
      R(params.N, stream),
      mask(params.N, stream),
      m(1, stream)
  {
  }

  void Run()
  {
    raft::update_device(labels_a.data(), params.labels_a.data(), params.N, stream);
    raft::update_device(labels_b.data(), params.labels_b.data(), params.N, stream);
    raft::update_device(expected.data(), params.expected.data(), params.N, stream);
    raft::update_device(mask.data(), reinterpret_cast<bool*>(params.mask.data()), params.N, stream);

    merge_labels(
      labels_a.data(), labels_b.data(), mask.data(), R.data(), m.data(), params.N, stream);

    cudaStreamSynchronize(stream);
    ASSERT_TRUE(raft::devArrMatch<Index_>(
      expected.data(), labels_a.data(), params.N, raft::Compare<Index_>()));
  }

 protected:
  MergeLabelsInputs<Index_> params;
  raft::handle_t handle;
  cudaStream_t stream = 0;
  rmm::device_uvector<Index_> labels_a, labels_b, expected, R;
  rmm::device_uvector<bool> mask, m;
};

using MergeLabelsTestI = MergeLabelsTest<int>;
TEST_P(MergeLabelsTestI, Result) { Run(); }

using MergeLabelsTestL = MergeLabelsTest<int64_t>;
TEST_P(MergeLabelsTestL, Result) { Run(); }

constexpr int MAX32     = std::numeric_limits<int>::max();
constexpr int64_t MAX64 = std::numeric_limits<int64_t>::max();

const std::vector<MergeLabelsInputs<int>> merge_inputs_32 = {
  {4, {1, 1, 3, MAX32}, {1, 3, 3, 1}, {1, 0, 1, 0}, {1, 1, 3, 1}},
  {5, {1, 2, 2, 2, 1}, {4, 2, 4, 4, 4}, {1, 1, 1, 1, 1}, {1, 1, 1, 1, 1}},
  {6, {1, 2, 1, 4, 5, MAX32}, {1, 2, MAX32, 4, 5, 4}, {1, 1, 0, 1, 1, 0}, {1, 2, 1, 4, 5, 4}},
  {6, {1, 2, 2, 2, 2, 6}, {1, 1, 1, 5, 5, 5}, {1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1}},
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
  {6, {1, 2, 1, 4, 5, MAX64}, {1, 2, MAX64, 4, 5, 4}, {1, 1, 0, 1, 1, 0}, {1, 2, 1, 4, 5, 4}},
  {6, {1, 2, 2, 2, 2, 6}, {1, 1, 1, 5, 5, 5}, {1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1}},
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

INSTANTIATE_TEST_CASE_P(MergeLabelsTests, MergeLabelsTestI, ::testing::ValuesIn(merge_inputs_32));
INSTANTIATE_TEST_CASE_P(MergeLabelsTests, MergeLabelsTestL, ::testing::ValuesIn(merge_inputs_64));

}  // namespace Label
}  // namespace MLCommon

/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

#include <cuml/common/utils.hpp>

#include <raft/core/interruptible.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <decoupled_lookback.cuh>
#include <gtest/gtest.h>

namespace MLCommon {

template <int TPB>
CUML_KERNEL void dlbTestKernel(void* workspace, int len, int* out)
{
  DecoupledLookBack<int> dlb(workspace);
  int count   = threadIdx.x == blockDim.x - 1 ? 1 : 0;
  auto prefix = dlb(count);
  if (threadIdx.x == blockDim.x - 1) out[blockIdx.x] = prefix;
}

void dlbTest(int len, int* out, cudaStream_t stream)
{
  constexpr int TPB    = 256;
  int nblks            = len;
  size_t workspaceSize = DecoupledLookBack<int>::computeWorkspaceSize(nblks);
  rmm::device_uvector<char> workspace(workspaceSize, stream);
  RAFT_CUDA_TRY(cudaMemset(workspace.data(), 0, workspace.size()));
  dlbTestKernel<TPB><<<nblks, TPB>>>(workspace.data(), len, out);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

struct DlbInputs {
  int len;
};

::std::ostream& operator<<(::std::ostream& os, const DlbInputs& dims) { return os; }

class DlbTest : public ::testing::TestWithParam<DlbInputs> {
 protected:
  DlbTest() : out(0, stream) {}

  void SetUp() override
  {
    RAFT_CUDA_TRY(cudaStreamCreate(&stream));

    params  = ::testing::TestWithParam<DlbInputs>::GetParam();
    int len = params.len;
    out.resize(len, stream);
    dlbTest(len, out.data(), stream);
  }

 protected:
  cudaStream_t stream = 0;
  DlbInputs params;
  rmm::device_uvector<int> out;
};

template <typename T, typename L>
::testing::AssertionResult devArrMatchCustom(const T* actual,
                                             size_t size,
                                             L eq_compare,
                                             cudaStream_t stream = 0)
{
  std::vector<T> act_h(size);
  raft::update_host<T>(&(act_h[0]), actual, size, stream);
  raft::interruptible::synchronize(stream);
  for (size_t i(0); i < size; ++i) {
    auto act      = act_h[i];
    auto expected = (T)i;
    if (!eq_compare(expected, act)) {
      return ::testing::AssertionFailure()
             << "actual=" << act << " != expected=" << expected << " @" << i;
    }
  }
  return ::testing::AssertionSuccess();
}

const std::vector<DlbInputs> inputs = {{4}, {16}, {64}, {256}, {2048}};
TEST_P(DlbTest, Result)
{
  ASSERT_TRUE(devArrMatchCustom(out.data(), params.len, MLCommon::Compare<int>()));
}
INSTANTIATE_TEST_CASE_P(DlbTests, DlbTest, ::testing::ValuesIn(inputs));

}  // end namespace MLCommon

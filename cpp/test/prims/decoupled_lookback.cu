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
#include <raft/cudart_utils.h>
#include <decoupled_lookback.cuh>
#include <rmm/device_uvector.hpp>
#include "test_utils.h"

namespace MLCommon {

template <int TPB>
__global__ void dlbTestKernel(void* workspace, int len, int* out)
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
  CUDA_CHECK(cudaMemset(workspace.data(), 0, workspace.size()));
  dlbTestKernel<TPB><<<nblks, TPB>>>(workspace.data(), len, out);
  CUDA_CHECK(cudaPeekAtLastError());
}

struct DlbInputs {
  int len;
};

::std::ostream& operator<<(::std::ostream& os, const DlbInputs& dims) { return os; }

class DlbTest : public ::testing::TestWithParam<DlbInputs> {
 protected:
  void SetUp() override
  {
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    params  = ::testing::TestWithParam<DlbInputs>::GetParam();
    int len = params.len;
    out     = std::make_unique<rmm::device_uvector<int>>(len, stream);
    dlbTest(len, out->data(), stream);
  }

 protected:
  DlbInputs params;
  std::unique_ptr<rmm::device_uvector<int>> out;
};

template <typename T, typename L>
::testing::AssertionResult devArrMatchCustom(const T* actual,
                                             size_t size,
                                             L eq_compare,
                                             cudaStream_t stream = 0)
{
  std::vector<T> act_h(size);
  raft::update_host<T>(&(act_h[0]), actual, size, stream);
  CUDA_CHECK(cudaStreamSynchronize(stream));
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
  ASSERT_TRUE(devArrMatchCustom(out->data(), params.len, raft::Compare<int>()));
}
INSTANTIATE_TEST_CASE_P(DlbTests, DlbTest, ::testing::ValuesIn(inputs));

}  // end namespace MLCommon

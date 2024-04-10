/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <common/grid_sync.cuh>

#include <cuml/common/utils.hpp>

#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

namespace MLCommon {

CUML_KERNEL void gridSyncTestKernel(void* workspace, int* out, SyncType type)
{
  GridSync gs(workspace, type, true);
  bool master;
  int updatePosition;
  if (type == ACROSS_ALL) {
    master = threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 &&
             blockIdx.y == 0 && blockIdx.z == 0;
    updatePosition = 0;
  } else {
    master         = threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0;
    updatePosition = blockIdx.y + blockIdx.z * gridDim.y;
  }
  if (master) {
    out[updatePosition] = 1;
    __threadfence();
  }
  gs.sync();
  int val = out[updatePosition];
  // make sure everybody has read the updated value!
  gs.sync();
  raft::myAtomicAdd(out + updatePosition, val);
}

struct GridSyncInputs {
  dim3 gridDim, blockDim;
  bool checkWorkspaceReuse;
  SyncType type;
};

void gridSyncTest(int* out, int* out1, const GridSyncInputs& params, cudaStream_t stream)
{
  size_t workspaceSize = GridSync::computeWorkspaceSize(params.gridDim, params.type, true);
  rmm::device_uvector<char> workspace(workspaceSize, stream);
  RAFT_CUDA_TRY(cudaMemset(workspace.data(), 0, workspace.size()));
  gridSyncTestKernel<<<params.gridDim, params.blockDim>>>(workspace.data(), out, params.type);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  if (params.checkWorkspaceReuse) {
    RAFT_CUDA_TRY(cudaDeviceSynchronize());
    gridSyncTestKernel<<<params.gridDim, params.blockDim>>>(workspace.data(), out1, params.type);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
}

::std::ostream& operator<<(::std::ostream& os, const GridSyncInputs& dims) { return os; }

class GridSyncTest : public ::testing::TestWithParam<GridSyncInputs> {
 protected:
  GridSyncTest() : out(0, stream), out1(0, stream) {}

  void SetUp() override
  {
    params     = ::testing::TestWithParam<GridSyncInputs>::GetParam();
    size_t len = computeOutLen();

    RAFT_CUDA_TRY(cudaStreamCreate(&stream));

    out.resize(len, stream);
    out1.resize(len, stream);
    gridSyncTest(out.data(), out1.data(), params, stream);
  }

  size_t computeOutLen() const
  {
    size_t len;
    if (params.type == ACROSS_ALL) {
      len = 1;
    } else {
      len = params.gridDim.y * params.gridDim.z;
    }
    return len;
  }

 protected:
  cudaStream_t stream = 0;
  GridSyncInputs params;
  rmm::device_uvector<int> out, out1;
};

const std::vector<GridSyncInputs> inputs = {
  {{2, 1, 1}, {32, 1, 1}, false, ACROSS_ALL}, {{2, 1, 1}, {32, 2, 1}, false, ACROSS_ALL},
  {{2, 1, 1}, {32, 2, 4}, false, ACROSS_ALL}, {{2, 1, 1}, {32, 1, 1}, true, ACROSS_ALL},
  {{2, 1, 1}, {32, 2, 1}, true, ACROSS_ALL},  {{2, 1, 1}, {32, 2, 4}, true, ACROSS_ALL},
  {{2, 1, 1}, {32, 1, 1}, false, ACROSS_X},   {{2, 2, 1}, {32, 1, 1}, false, ACROSS_X},
  {{2, 2, 2}, {32, 1, 1}, false, ACROSS_X},   {{2, 1, 1}, {32, 2, 1}, false, ACROSS_X},
  {{2, 2, 1}, {32, 2, 1}, false, ACROSS_X},   {{2, 2, 2}, {32, 2, 1}, false, ACROSS_X},
  {{2, 1, 1}, {32, 2, 4}, false, ACROSS_X},   {{2, 2, 1}, {32, 2, 4}, false, ACROSS_X},
  {{2, 2, 2}, {32, 2, 4}, false, ACROSS_X},   {{32, 256, 1}, {1, 1, 1}, false, ACROSS_X},
  {{2, 1, 1}, {32, 1, 1}, true, ACROSS_X},    {{2, 2, 1}, {32, 1, 1}, true, ACROSS_X},
  {{2, 2, 2}, {32, 1, 1}, true, ACROSS_X},    {{2, 1, 1}, {32, 2, 1}, true, ACROSS_X},
  {{2, 2, 1}, {32, 2, 1}, true, ACROSS_X},    {{2, 2, 2}, {32, 2, 1}, true, ACROSS_X},
  {{2, 1, 1}, {32, 2, 4}, true, ACROSS_X},    {{2, 2, 1}, {32, 2, 4}, true, ACROSS_X},
  {{2, 2, 2}, {32, 2, 4}, true, ACROSS_X},    {{32, 256, 1}, {1, 1, 1}, true, ACROSS_X}};
TEST_P(GridSyncTest, Result)
{
  size_t len = computeOutLen();
  // number of blocks raft::myAtomicAdd'ing the same location
  int nblks    = params.type == ACROSS_X ? params.gridDim.x
                                         : params.gridDim.x * params.gridDim.y * params.gridDim.z;
  int nthreads = params.blockDim.x * params.blockDim.y * params.blockDim.z;
  int expected = (nblks * nthreads) + 1;
  ASSERT_TRUE(MLCommon::devArrMatch(expected, out.data(), len, MLCommon::Compare<int>()));
  if (params.checkWorkspaceReuse) {
    ASSERT_TRUE(MLCommon::devArrMatch(expected, out1.data(), len, MLCommon::Compare<int>()));
  }
}
INSTANTIATE_TEST_CASE_P(GridSyncTests, GridSyncTest, ::testing::ValuesIn(inputs));

}  // end namespace MLCommon

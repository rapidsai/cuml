/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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
#include "cuda_utils.h"
#include "test_utils.h"
#include "common/grid_sync.h"

namespace MLCommon {

__global__ void gridSyncTestKernel(void* workspace, int* out, SyncType type) {
    GridSync gs(workspace, type);
    if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 &&
       blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        out[0] = 1;
        __threadfence();
    }
    gs.sync();
    int val = out[0];
    // make sure everybody has read the updated value!
    gs.sync();
    atomicAdd(out, val);
}

struct GridSyncInputs {
    dim3 gridDim, blockDim;
    bool checkWorkspaceReuse;
    SyncType type;
};

void gridSyncTest(int* out, int* out1, const GridSyncInputs& params) {
    size_t workspaceSize = GridSync::computeWorkspaceSize(params.gridDim,
                                                          params.type);
    char* workspace;
    allocate(workspace, workspaceSize);
    CUDA_CHECK(cudaMemset(workspace, 0, workspaceSize));
    gridSyncTestKernel<<<params.gridDim, params.blockDim>>>(workspace, out,
                                                            params.type);
    CUDA_CHECK(cudaPeekAtLastError());
    if(params.checkWorkspaceReuse) {
        CUDA_CHECK(cudaDeviceSynchronize());
        gridSyncTestKernel<<<params.gridDim, params.blockDim>>>(workspace, out1,
                                                                params.type);
        CUDA_CHECK(cudaPeekAtLastError());
    }
    CUDA_CHECK(cudaFree(workspace));
}

::std::ostream &operator<<(::std::ostream &os, const GridSyncInputs &dims) {
    return os;
}

class GridSyncTest : public ::testing::TestWithParam<GridSyncInputs> {
protected:
  void SetUp() override {
    params = ::testing::TestWithParam<GridSyncInputs>::GetParam();
    allocate(out, 1);
    allocate(out1, 1);
    gridSyncTest(out, out1, params);
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(out1));
  }

protected:
  GridSyncInputs params;
  int *out, *out1;
};


template <typename L>
::testing::AssertionResult devArrMatchSingle(const int *actual, int expected,
                                             L eq_compare) {
  int act_h;
  updateHost(&act_h, actual, 1);
  if (!eq_compare(expected, act_h)) {
    return ::testing::AssertionFailure()
        << "actual=" << act_h << " != expected=" << expected;
  }
  return ::testing::AssertionSuccess();
}


const std::vector<GridSyncInputs> inputs = {
  {{2, 1, 1}, {32, 1, 1}, false, ACROSS_ALL},
  {{2, 1, 1}, {32, 2, 1}, false, ACROSS_ALL},
  {{2, 1, 1}, {32, 2, 4}, false, ACROSS_ALL},
  {{2, 1, 1}, {32, 1, 1}, true, ACROSS_ALL},
  {{2, 1, 1}, {32, 2, 1}, true, ACROSS_ALL},
  {{2, 1, 1}, {32, 2, 4}, true, ACROSS_ALL},
  {{2, 1, 1}, {32, 1, 1}, false, ACROSS_X},
  {{2, 2, 1}, {32, 1, 1}, false, ACROSS_X},
  {{2, 2, 2}, {32, 1, 1}, false, ACROSS_X},
  {{2, 1, 1}, {32, 2, 1}, false, ACROSS_X},
  {{2, 2, 1}, {32, 2, 1}, false, ACROSS_X},
  {{2, 2, 2}, {32, 2, 1}, false, ACROSS_X},
  {{2, 1, 1}, {32, 2, 4}, false, ACROSS_X},
  {{2, 2, 1}, {32, 2, 4}, false, ACROSS_X},
  {{2, 2, 2}, {32, 2, 4}, false, ACROSS_X},
  {{2, 1, 1}, {32, 1, 1}, true, ACROSS_X},
  {{2, 2, 1}, {32, 1, 1}, true, ACROSS_X},
  {{2, 2, 2}, {32, 1, 1}, true, ACROSS_X},
  {{2, 1, 1}, {32, 2, 1}, true, ACROSS_X},
  {{2, 2, 1}, {32, 2, 1}, true, ACROSS_X},
  {{2, 2, 2}, {32, 2, 1}, true, ACROSS_X},
  {{2, 1, 1}, {32, 2, 4}, true, ACROSS_X},
  {{2, 2, 1}, {32, 2, 4}, true, ACROSS_X},
  {{2, 2, 2}, {32, 2, 4}, true, ACROSS_X}};
TEST_P(GridSyncTest, Result) {
  int nblks = params.gridDim.x * params.gridDim.y * params.gridDim.z;
  int nthreads = params.blockDim.x * params.blockDim.y * params.blockDim.z;
  int expected = (nblks * nthreads) + 1;
  ASSERT_TRUE(devArrMatchSingle(out, expected, Compare<int>()));
  if(params.checkWorkspaceReuse) {
    ASSERT_TRUE(devArrMatchSingle(out1, expected, Compare<int>()));
  }
}
INSTANTIATE_TEST_CASE_P(GridSyncTests, GridSyncTest, ::testing::ValuesIn(inputs));

} // end namespace MLCommon

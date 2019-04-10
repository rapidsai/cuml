/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

__global__ void gridSyncTestKernel(void* workspace1, void* workspace2, int* out,
                                   SyncType type) {
    GridSync gs1(workspace1, type);
    GridSync gs2(workspace2, type);
    bool master;
    int updatePosition;
    if(type == ACROSS_ALL) {
        master = threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 &&
            blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0;
        updatePosition = 0;
    } else {
        master = threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 &&
            blockIdx.x == 0;
        updatePosition = blockIdx.y + blockIdx.z * gridDim.y;
    }
    if(master) {
        out[updatePosition] = 1;
        __threadfence();
    }
    gs1.sync();
    int val = out[updatePosition];
    // make sure everybody has read the updated value!
    gs2.sync();
    atomicAdd(out+updatePosition, val);
}

struct GridSyncInputs {
    dim3 gridDim, blockDim;
    bool checkWorkspaceReuse;
    SyncType type;
};

void gridSyncTest(int* out, int* out1, const GridSyncInputs& params) {
    size_t workspaceSize = GridSync::computeWorkspaceSize(params.gridDim,
                                                          params.type);
    char *workspace1, * workspace2;
    allocate(workspace1, workspaceSize);
    allocate(workspace2, workspaceSize);
    CUDA_CHECK(cudaMemset(workspace1, 0, workspaceSize));
    CUDA_CHECK(cudaMemset(workspace2, 0, workspaceSize));
    gridSyncTestKernel<<<params.gridDim, params.blockDim>>>(workspace1,
                                                            workspace2,
                                                            out, params.type);
    CUDA_CHECK(cudaPeekAtLastError());
    if(params.checkWorkspaceReuse) {
        CUDA_CHECK(cudaDeviceSynchronize());
        gridSyncTestKernel<<<params.gridDim, params.blockDim>>>(workspace1,
                                                                workspace2,
                                                                out1,
                                                                params.type);
        CUDA_CHECK(cudaPeekAtLastError());
    }
    CUDA_CHECK(cudaFree(workspace1));
    CUDA_CHECK(cudaFree(workspace2));
}

::std::ostream &operator<<(::std::ostream &os, const GridSyncInputs &dims) {
    return os;
}

class GridSyncTest : public ::testing::TestWithParam<GridSyncInputs> {
protected:
  void SetUp() override {
    params = ::testing::TestWithParam<GridSyncInputs>::GetParam();
    size_t len = computeOutLen();
    allocate(out, len);
    allocate(out1, len);
    gridSyncTest(out, out1, params);
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(out1));
  }

  size_t computeOutLen() const {
    size_t len;
    if(params.type == ACROSS_ALL) {
      len = 1;
    } else {
      len = params.gridDim.y * params.gridDim.z;
    }
    return len;
  }

protected:
  GridSyncInputs params;
  int *out, *out1;
};


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
  {{32, 256, 1}, {1, 1, 1}, false, ACROSS_X},
  {{2, 1, 1}, {32, 1, 1}, true, ACROSS_X},
  {{2, 2, 1}, {32, 1, 1}, true, ACROSS_X},
  {{2, 2, 2}, {32, 1, 1}, true, ACROSS_X},
  {{2, 1, 1}, {32, 2, 1}, true, ACROSS_X},
  {{2, 2, 1}, {32, 2, 1}, true, ACROSS_X},
  {{2, 2, 2}, {32, 2, 1}, true, ACROSS_X},
  {{2, 1, 1}, {32, 2, 4}, true, ACROSS_X},
  {{2, 2, 1}, {32, 2, 4}, true, ACROSS_X},
  {{2, 2, 2}, {32, 2, 4}, true, ACROSS_X},
  {{32, 256, 1}, {1, 1, 1}, true, ACROSS_X}};
TEST_P(GridSyncTest, Result) {
  size_t len = computeOutLen();
  // number of blocks atomicAdd'ing the same location
  int nblks = params.type == ACROSS_X?
    params.gridDim.x : params.gridDim.x * params.gridDim.y * params.gridDim.z;
  int nthreads = params.blockDim.x * params.blockDim.y * params.blockDim.z;
  int expected = (nblks * nthreads) + 1;
  ASSERT_TRUE(devArrMatch(expected, out, len, Compare<int>()));
  if(params.checkWorkspaceReuse) {
    ASSERT_TRUE(devArrMatch(expected, out1, len, Compare<int>()));
  }
}
INSTANTIATE_TEST_CASE_P(GridSyncTests, GridSyncTest, ::testing::ValuesIn(inputs));

} // end namespace MLCommon

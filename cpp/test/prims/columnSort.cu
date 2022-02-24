/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include <algorithm>
#include <gtest/gtest.h>
#include <numeric>
#include <raft/cudart_utils.h>
#include <rmm/device_uvector.hpp>
#include <selection/columnWiseSort.cuh>

namespace MLCommon {
namespace Selection {

template <typename T>
std::vector<int>* sort_indexes(const std::vector<T>& v)
{
  // initialize original index locations
  std::vector<int>* idx = new std::vector<int>(v.size());
  std::iota((*idx).begin(), (*idx).end(), 0);

  // sort indexes based on comparing values in v
  std::sort((*idx).begin(), (*idx).end(), [&v](int i1, int i2) { return v[i1] < v[i2]; });
  return idx;
}

template <typename T>
struct columnSort {
  T tolerance;
  int n_row;
  int n_col;
  bool testKeys;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const columnSort<T>& dims)
{
  return os;
}

template <typename T>
class ColumnSort : public ::testing::TestWithParam<columnSort<T>> {
 protected:
  ColumnSort()
    : keyIn(0, stream),
      keySorted(0, stream),
      keySortGolden(0, stream),
      valueOut(0, stream),
      goldenValOut(0, stream),
      workspacePtr(0, stream)
  {
  }

  void SetUp() override
  {
    params  = ::testing::TestWithParam<columnSort<T>>::GetParam();
    int len = params.n_row * params.n_col;
    RAFT_CUDA_TRY(cudaStreamCreate(&stream));
    keyIn.resize(len, stream);
    valueOut.resize(len, stream);
    goldenValOut.resize(len, stream);
    if (params.testKeys) {
      keySorted.resize(len, stream);
      keySortGolden.resize(len, stream);
    }

    std::vector<T> vals(len);
    std::vector<int> cValGolden(len);
    std::iota(vals.begin(), vals.end(),
              1.0f);  // will have to change input param type
    std::random_shuffle(vals.begin(), vals.end());

    std::vector<T> cKeyGolden(len);

    for (int i = 0; i < params.n_row; i++) {
      std::vector<T> tmp(vals.begin() + i * params.n_col, vals.begin() + (i + 1) * params.n_col);
      auto cpuOut = sort_indexes(tmp);
      std::copy((*cpuOut).begin(), (*cpuOut).end(), cValGolden.begin() + i * params.n_col);
      delete cpuOut;

      if (params.testKeys) {
        std::sort(tmp.begin(), tmp.end());
        std::copy(tmp.begin(), tmp.end(), cKeyGolden.begin() + i * params.n_col);
      }
    }

    raft::update_device(keyIn.data(), &vals[0], len, stream);
    raft::update_device(goldenValOut.data(), &cValGolden[0], len, stream);

    if (params.testKeys) raft::update_device(keySortGolden.data(), &cKeyGolden[0], len, stream);

    bool needWorkspace   = false;
    size_t workspaceSize = 0;
    // Remove this branch once the implementation of descending sort is fixed.
    sortColumnsPerRow(keyIn.data(),
                      valueOut.data(),
                      params.n_row,
                      params.n_col,
                      needWorkspace,
                      NULL,
                      workspaceSize,
                      stream,
                      keySorted.data());
    if (needWorkspace) {
      workspacePtr.resize(workspaceSize, stream);
      sortColumnsPerRow(keyIn.data(),
                        valueOut.data(),
                        params.n_row,
                        params.n_col,
                        needWorkspace,
                        workspacePtr.data(),
                        workspaceSize,
                        stream,
                        keySorted.data());
    }
    RAFT_CUDA_TRY(cudaStreamDestroy(stream));
  }

 protected:
  cudaStream_t stream = 0;
  columnSort<T> params;
  rmm::device_uvector<T> keyIn, keySorted, keySortGolden;
  rmm::device_uvector<int> valueOut, goldenValOut;  // valueOut are indexes
  rmm::device_uvector<char> workspacePtr;
};

const std::vector<columnSort<float>> inputsf1 = {{0.000001f, 503, 2000, false},
                                                 {0.000001f, 113, 20000, true},
                                                 {0.000001f, 503, 2000, false},
                                                 {0.000001f, 113, 20000, true}};

typedef ColumnSort<float> ColumnSortF;
TEST_P(ColumnSortF, Result)
{
  // Remove this condition once the implementation of of descending sort is
  // fixed.
  ASSERT_TRUE(devArrMatch(valueOut.data(),
                          goldenValOut.data(),
                          params.n_row * params.n_col,
                          raft::CompareApprox<float>(params.tolerance)));
  if (params.testKeys) {
    ASSERT_TRUE(devArrMatch(keySorted.data(),
                            keySortGolden.data(),
                            params.n_row * params.n_col,
                            raft::CompareApprox<float>(params.tolerance)));
  }
}

INSTANTIATE_TEST_CASE_P(ColumnSortTests, ColumnSortF, ::testing::ValuesIn(inputsf1));

}  // end namespace Selection
}  // end namespace MLCommon

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
#include "selection/columnWiseSort.h"
#include "test_utils.h"
#include <numeric>
#include <algorithm>

namespace MLCommon {
namespace Selection {

template <typename T>
  std::vector<int>* sort_indexes(const vector<T> &v) {
  // initialize original index locations
  std::vector<int> *idx = new std::vector<int>(v.size());
  std::iota((*idx).begin(), (*idx).end(), 0);

  // sort indexes based on comparing values in v
  std::sort((*idx).begin(), (*idx).end(), [&v](int i1, int i2) {return v[i1] < v[i2]; });
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
::std::ostream &operator<<(::std::ostream &os, const columnSort<T> &dims) {
  return os;
}

template <typename T>
class ColumnSort : public ::testing::TestWithParam<columnSort<T>> {
protected:
  void SetUp() override {
    params = ::testing::TestWithParam<columnSort<T>>::GetParam();
    int len = params.n_row * params.n_col;
    allocate(keyIn, len);
    allocate(valueOut, len);
    allocate(goldenValOut, len);
    if (params.testKeys) {
      allocate(keySorted, len);
      allocate(keySortGolden, len);
    }

    std::vector<T> vals(len);
    std::vector<int> cValGolden(len);
    std::iota(vals.begin(), vals.end(), 1.0f);  //will have to change input param type
    std::random_shuffle(vals.begin(), vals.end());

    std::vector<T> cKeyGolden(len);

    for (int i = 0; i < params.n_row; i++) {
      std::vector<T> tmp(vals.begin() + i*params.n_col, vals.begin() + (i+1)*params.n_col);
      auto cpuOut = sort_indexes(tmp);
      std::copy((*cpuOut).begin(), (*cpuOut).end(), cValGolden.begin() + i*params.n_col);
      delete cpuOut;

      if (params.testKeys) {
        std::sort(tmp.begin(), tmp.end());
        std::copy(tmp.begin(), tmp.end(), cKeyGolden.begin() + i*params.n_col);
      }
    }

    updateDevice(keyIn, &vals[0], len);
    updateDevice(goldenValOut, &cValGolden[0], len);

    if (params.testKeys)
      updateDevice(keySortGolden, &cKeyGolden[0], len);

    bool isColumnMajor = false;
    bool needWorkspace = false;
    size_t workspaceSize = 0;
    sortColumnsPerRow(keyIn, valueOut, params.n_row, params.n_col, isColumnMajor,
                        needWorkspace, NULL, workspaceSize, keySorted);
    if (needWorkspace) {
      allocate(workspacePtr, workspaceSize);
      sortColumnsPerRow(keyIn, valueOut, params.n_row, params.n_col, isColumnMajor, needWorkspace, 
                            workspacePtr, workspaceSize, keySorted);
    }    
}

  void TearDown() override {
    CUDA_CHECK(cudaFree(keyIn));
    CUDA_CHECK(cudaFree(valueOut));
    CUDA_CHECK(cudaFree(goldenValOut));
    if (params.testKeys) {
      CUDA_CHECK(cudaFree(keySorted));
      CUDA_CHECK(cudaFree(keySortGolden));
    }
    if (!workspacePtr)
      CUDA_CHECK(cudaFree(workspacePtr));
}

protected:
  columnSort<T> params;
  T *keyIn;
  T *keySorted = NULL;
  T *keySortGolden = NULL;
  int *valueOut, *goldenValOut;    // valueOut are indexes
  char *workspacePtr=NULL;
};

const std::vector<columnSort<float>> inputsf1 = {
  {0.000001f, 503, 2000, false},
  {0.000001f, 128, 20000, false},
  {0.000001f, 128, 100000, false},
  {0.000001f, 503, 2000, true},
  {0.000001f, 128, 20000, true},
  {0.000001f, 128, 100000, true}};

typedef ColumnSort<float> ColumnSortF;
TEST_P(ColumnSortF, Result) {
  ASSERT_TRUE(devArrMatch(valueOut, goldenValOut, params.n_row * params.n_col,
                  CompareApprox<float>(params.tolerance)));
  if (params.testKeys) {
    ASSERT_TRUE(devArrMatch(keySorted, keySortGolden, params.n_row * params.n_col,
      CompareApprox<float>(params.tolerance)));
  }
}

INSTANTIATE_TEST_CASE_P(ColumnSortTests, ColumnSortF,
                ::testing::ValuesIn(inputsf1));

} // end namespace Selection
} // end namespace MLCommon

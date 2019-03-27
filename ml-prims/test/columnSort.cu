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
  std::vector<int> sort_indexes(const vector<T> &v) {
  // initialize original index locations
  std::vector<int> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  std::sort(idx.begin(), idx.end(), [&v](int i1, int i2) {return v[i1] < v[i2]; });
  return idx;
}

template <typename T>
struct columnSort {
  T tolerance;
  int n_row;
  int n_col;
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
    allocate(in, len);
    allocate(out, len);
    allocate(golden, len);

    std::vector<T> vals(len);
    std::vector<int> cGolden(len);
    std::iota(vals.begin(), vals.end(), 1.0f);  //will have to change input param type
    std::random_shuffle(vals.begin(), vals.end());

    for (int i = 0; i < params.n_row; i++) {
      std::vector<T> tmp(vals.begin() + i*params.n_col, vals.begin() + (i+1)*params.n_col);
      auto cpuOut = sort_indexes(tmp);

      std::copy(cpuOut.begin(), cpuOut.end(), cGolden.begin() + i*params.n_col);
    }
    
    updateDevice(in, &vals[0], len);
    updateDevice(golden, &cGolden[0], len);

    sortColumnsPerRow(in, out, params.n_row, params.n_col, false, false, NULL, 0);
    //CUDA_CHECK(cudaDeviceSynchronize());

    // T *dbg = (T *)malloc(len * sizeof(T));
    // CUDA_CHECK(cudaMemcpy(dbg, in, len*sizeof(T), cudaMemcpyDeviceToHost));
    // int *dbg = (int *)malloc(len * sizeof(int));
    // CUDA_CHECK(cudaMemcpy(dbg, out, len*sizeof(int), cudaMemcpyDeviceToHost));
    // for (int i = 0; i < params.n_col; i++)
    //   std::cout << dbg[i] << " " << cGolden[i] << std::endl;
}

  void TearDown() override {
    CUDA_CHECK(cudaFree(in));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(golden));
}

protected:
  columnSort<T> params;
  T *in;
  int *out, *golden;
};

const std::vector<columnSort<float>> inputsf2 = {{0.000001f, 1024, 5000}};

typedef ColumnSort<float> ColumnSortF;
TEST_P(ColumnSortF, Result) {
ASSERT_TRUE(devArrMatch(out, golden, params.n_row * params.n_col,
                CompareApprox<float>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(ColumnSortTests, ColumnSortF,
                ::testing::ValuesIn(inputsf2));


} // end namespace Selection
} // end namespace MLCommon

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
#include <set>
#include <vector>
#include "cuda_utils.h"
#include "random/rng.h"
#include "test_utils.h"

namespace MLCommon {
namespace Random {

// Terminology:
// SWoR - Sample Without Replacement

template <typename T>
struct SWoRInputs {
  int len, sampledLen;
  int largeWeightIndex;
  T largeWeight;
  GeneratorType gtype;
  unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const SWoRInputs<T>& dims) {
  return os;
}

template <typename T>
class SWoRTest : public ::testing::TestWithParam<SWoRInputs<T>> {
 protected:
  void SetUp() override {
    params = ::testing::TestWithParam<SWoRInputs<T>>::GetParam();
    CUDA_CHECK(cudaStreamCreate(&stream));
    allocator.reset(new defaultDeviceAllocator);
    Rng r(params.seed, params.gtype);
    allocate(in, params.len);
    allocate(wts, params.len);
    allocate(out, params.sampledLen);
    allocate(outIdx, params.sampledLen);
    h_outIdx.resize(params.sampledLen);
    r.uniform(in, params.len, T(-1.0), T(1.0), stream);
    r.uniform(wts, params.len, T(1.0), T(2.0), stream);
    if (params.largeWeightIndex >= 0) {
      updateDevice(wts + params.largeWeightIndex, &params.largeWeight, 1,
                   stream);
    }
    r.sampleWithoutReplacement(out, outIdx, in, wts, params.sampledLen,
                               params.len, allocator, stream);
    updateHost(&(h_outIdx[0]), outIdx, params.sampledLen, stream);
  }

  void TearDown() override {
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(in));
    CUDA_CHECK(cudaFree(wts));
    CUDA_CHECK(cudaFree(out));
    CUDA_CHECK(cudaFree(outIdx));
  }

 protected:
  SWoRInputs<T> params;
  T *in, *out, *wts;
  int* outIdx;
  std::vector<int> h_outIdx;
  cudaStream_t stream;
  std::shared_ptr<deviceAllocator> allocator;
};

typedef SWoRTest<float> SWoRTestF;
const std::vector<SWoRInputs<float>> inputsf = {
  {1024, 512, -1, 0.f, GenPhilox, 1234ULL},
  {1024, 1024, -1, 0.f, GenPhilox, 1234ULL},
  {1024, 512 + 1, -1, 0.f, GenPhilox, 1234ULL},
  {1024, 1024 - 1, -1, 0.f, GenPhilox, 1234ULL},
  {1024, 512 + 2, -1, 0.f, GenPhilox, 1234ULL},
  {1024, 1024 - 2, -1, 0.f, GenPhilox, 1234ULL},
  {1024 + 1, 512, -1, 0.f, GenPhilox, 1234ULL},
  {1024 + 1, 1024, -1, 0.f, GenPhilox, 1234ULL},
  {1024 + 1, 512 + 1, -1, 0.f, GenPhilox, 1234ULL},
  {1024 + 1, 1024 + 1, -1, 0.f, GenPhilox, 1234ULL},
  {1024 + 1, 512 + 2, -1, 0.f, GenPhilox, 1234ULL},
  {1024 + 1, 1024 - 2, -1, 0.f, GenPhilox, 1234ULL},
  {1024 + 2, 512, -1, 0.f, GenPhilox, 1234ULL},
  {1024 + 2, 1024, -1, 0.f, GenPhilox, 1234ULL},
  {1024 + 2, 512 + 1, -1, 0.f, GenPhilox, 1234ULL},
  {1024 + 2, 1024 + 1, -1, 0.f, GenPhilox, 1234ULL},
  {1024 + 2, 512 + 2, -1, 0.f, GenPhilox, 1234ULL},
  {1024 + 2, 1024 + 2, -1, 0.f, GenPhilox, 1234ULL},
  {1024, 512, 10, 100000.f, GenPhilox, 1234ULL},

  {1024, 512, -1, 0.f, GenTaps, 1234ULL},
  {1024, 1024, -1, 0.f, GenTaps, 1234ULL},
  {1024, 512 + 1, -1, 0.f, GenTaps, 1234ULL},
  {1024, 1024 - 1, -1, 0.f, GenTaps, 1234ULL},
  {1024, 512 + 2, -1, 0.f, GenTaps, 1234ULL},
  {1024, 1024 - 2, -1, 0.f, GenTaps, 1234ULL},
  {1024 + 1, 512, -1, 0.f, GenTaps, 1234ULL},
  {1024 + 1, 1024, -1, 0.f, GenTaps, 1234ULL},
  {1024 + 1, 512 + 1, -1, 0.f, GenTaps, 1234ULL},
  {1024 + 1, 1024 + 1, -1, 0.f, GenTaps, 1234ULL},
  {1024 + 1, 512 + 2, -1, 0.f, GenTaps, 1234ULL},
  {1024 + 1, 1024 - 2, -1, 0.f, GenTaps, 1234ULL},
  {1024 + 2, 512, -1, 0.f, GenTaps, 1234ULL},
  {1024 + 2, 1024, -1, 0.f, GenTaps, 1234ULL},
  {1024 + 2, 512 + 1, -1, 0.f, GenTaps, 1234ULL},
  {1024 + 2, 1024 + 1, -1, 0.f, GenTaps, 1234ULL},
  {1024 + 2, 512 + 2, -1, 0.f, GenTaps, 1234ULL},
  {1024 + 2, 1024 + 2, -1, 0.f, GenTaps, 1234ULL},
  {1024, 512, 10, 100000.f, GenTaps, 1234ULL},

  {1024, 512, -1, 0.f, GenKiss99, 1234ULL},
  {1024, 1024, -1, 0.f, GenKiss99, 1234ULL},
  {1024, 512 + 1, -1, 0.f, GenKiss99, 1234ULL},
  {1024, 1024 - 1, -1, 0.f, GenKiss99, 1234ULL},
  {1024, 512 + 2, -1, 0.f, GenKiss99, 1234ULL},
  {1024, 1024 - 2, -1, 0.f, GenKiss99, 1234ULL},
  {1024 + 1, 512, -1, 0.f, GenKiss99, 1234ULL},
  {1024 + 1, 1024, -1, 0.f, GenKiss99, 1234ULL},
  {1024 + 1, 512 + 1, -1, 0.f, GenKiss99, 1234ULL},
  {1024 + 1, 1024 + 1, -1, 0.f, GenKiss99, 1234ULL},
  {1024 + 1, 512 + 2, -1, 0.f, GenKiss99, 1234ULL},
  {1024 + 1, 1024 - 2, -1, 0.f, GenKiss99, 1234ULL},
  {1024 + 2, 512, -1, 0.f, GenKiss99, 1234ULL},
  {1024 + 2, 1024, -1, 0.f, GenKiss99, 1234ULL},
  {1024 + 2, 512 + 1, -1, 0.f, GenKiss99, 1234ULL},
  {1024 + 2, 1024 + 1, -1, 0.f, GenKiss99, 1234ULL},
  {1024 + 2, 512 + 2, -1, 0.f, GenKiss99, 1234ULL},
  {1024 + 2, 1024 + 2, -1, 0.f, GenKiss99, 1234ULL},
  {1024, 512, 10, 100000.f, GenKiss99, 1234ULL},
};

TEST_P(SWoRTestF, Result) {
  std::set<int> occurence;
  for (int i = 0; i < params.sampledLen; ++i) {
    auto val = h_outIdx[i];
    // indices must be in the given range
    ASSERT_TRUE(0 <= val && val < params.len)
      << "out-of-range index @i=" << i << " val=" << val
      << " sampledLen=" << params.sampledLen;
    // indices should not repeat
    ASSERT_TRUE(occurence.find(val) == occurence.end())
      << "repeated index @i=" << i << " idx=" << val;
    occurence.insert(val);
  }
  // if there's a skewed distribution, the top index should correspond to the
  // particular item with a large weight
  if (params.largeWeightIndex >= 0) {
    ASSERT_EQ(h_outIdx[0], params.largeWeightIndex);
  }
}
INSTANTIATE_TEST_CASE_P(SWoRTests, SWoRTestF, ::testing::ValuesIn(inputsf));

typedef SWoRTest<double> SWoRTestD;
const std::vector<SWoRInputs<double>> inputsd = {
  {1024, 512, -1, 0.0, GenPhilox, 1234ULL},
  {1024, 1024, -1, 0.0, GenPhilox, 1234ULL},
  {1024, 512 + 1, -1, 0.0, GenPhilox, 1234ULL},
  {1024, 1024 - 1, -1, 0.0, GenPhilox, 1234ULL},
  {1024, 512 + 2, -1, 0.0, GenPhilox, 1234ULL},
  {1024, 1024 - 2, -1, 0.0, GenPhilox, 1234ULL},
  {1024 + 1, 512, -1, 0.0, GenPhilox, 1234ULL},
  {1024 + 1, 1024, -1, 0.0, GenPhilox, 1234ULL},
  {1024 + 1, 512 + 1, -1, 0.0, GenPhilox, 1234ULL},
  {1024 + 1, 1024 + 1, -1, 0.0, GenPhilox, 1234ULL},
  {1024 + 1, 512 + 2, -1, 0.0, GenPhilox, 1234ULL},
  {1024 + 1, 1024 - 2, -1, 0.0, GenPhilox, 1234ULL},
  {1024 + 2, 512, -1, 0.0, GenPhilox, 1234ULL},
  {1024 + 2, 1024, -1, 0.0, GenPhilox, 1234ULL},
  {1024 + 2, 512 + 1, -1, 0.0, GenPhilox, 1234ULL},
  {1024 + 2, 1024 + 1, -1, 0.0, GenPhilox, 1234ULL},
  {1024 + 2, 512 + 2, -1, 0.0, GenPhilox, 1234ULL},
  {1024 + 2, 1024 + 2, -1, 0.0, GenPhilox, 1234ULL},
  {1024, 512, 10, 100000.0, GenPhilox, 1234ULL},

  {1024, 512, -1, 0.0, GenTaps, 1234ULL},
  {1024, 1024, -1, 0.0, GenTaps, 1234ULL},
  {1024, 512 + 1, -1, 0.0, GenTaps, 1234ULL},
  {1024, 1024 - 1, -1, 0.0, GenTaps, 1234ULL},
  {1024, 512 + 2, -1, 0.0, GenTaps, 1234ULL},
  {1024, 1024 - 2, -1, 0.0, GenTaps, 1234ULL},
  {1024 + 1, 512, -1, 0.0, GenTaps, 1234ULL},
  {1024 + 1, 1024, -1, 0.0, GenTaps, 1234ULL},
  {1024 + 1, 512 + 1, -1, 0.0, GenTaps, 1234ULL},
  {1024 + 1, 1024 + 1, -1, 0.0, GenTaps, 1234ULL},
  {1024 + 1, 512 + 2, -1, 0.0, GenTaps, 1234ULL},
  {1024 + 1, 1024 - 2, -1, 0.0, GenTaps, 1234ULL},
  {1024 + 2, 512, -1, 0.0, GenTaps, 1234ULL},
  {1024 + 2, 1024, -1, 0.0, GenTaps, 1234ULL},
  {1024 + 2, 512 + 1, -1, 0.0, GenTaps, 1234ULL},
  {1024 + 2, 1024 + 1, -1, 0.0, GenTaps, 1234ULL},
  {1024 + 2, 512 + 2, -1, 0.0, GenTaps, 1234ULL},
  {1024 + 2, 1024 + 2, -1, 0.0, GenTaps, 1234ULL},
  {1024, 512, 10, 100000.0, GenTaps, 1234ULL},

  {1024, 512, -1, 0.0, GenKiss99, 1234ULL},
  {1024, 1024, -1, 0.0, GenKiss99, 1234ULL},
  {1024, 512 + 1, -1, 0.0, GenKiss99, 1234ULL},
  {1024, 1024 - 1, -1, 0.0, GenKiss99, 1234ULL},
  {1024, 512 + 2, -1, 0.0, GenKiss99, 1234ULL},
  {1024, 1024 - 2, -1, 0.0, GenKiss99, 1234ULL},
  {1024 + 1, 512, -1, 0.0, GenKiss99, 1234ULL},
  {1024 + 1, 1024, -1, 0.0, GenKiss99, 1234ULL},
  {1024 + 1, 512 + 1, -1, 0.0, GenKiss99, 1234ULL},
  {1024 + 1, 1024 + 1, -1, 0.0, GenKiss99, 1234ULL},
  {1024 + 1, 512 + 2, -1, 0.0, GenKiss99, 1234ULL},
  {1024 + 1, 1024 - 2, -1, 0.0, GenKiss99, 1234ULL},
  {1024 + 2, 512, -1, 0.0, GenKiss99, 1234ULL},
  {1024 + 2, 1024, -1, 0.0, GenKiss99, 1234ULL},
  {1024 + 2, 512 + 1, -1, 0.0, GenKiss99, 1234ULL},
  {1024 + 2, 1024 + 1, -1, 0.0, GenKiss99, 1234ULL},
  {1024 + 2, 512 + 2, -1, 0.0, GenKiss99, 1234ULL},
  {1024 + 2, 1024 + 2, -1, 0.0, GenKiss99, 1234ULL},
  {1024, 512, 10, 100000.0, GenKiss99, 1234ULL},
};

TEST_P(SWoRTestD, Result) {
  std::set<int> occurence;
  for (int i = 0; i < params.sampledLen; ++i) {
    auto val = h_outIdx[i];
    // indices must be in the given range
    ASSERT_TRUE(0 <= val && val < params.len)
      << "out-of-range index @i=" << i << " val=" << val
      << " sampledLen=" << params.sampledLen;
    // indices should not repeat
    ASSERT_TRUE(occurence.find(val) == occurence.end())
      << "repeated index @i=" << i << " idx=" << val;
    occurence.insert(val);
  }
  // if there's a skewed distribution, the top index should correspond to the
  // particular item with a large weight
  if (params.largeWeightIndex >= 0) {
    ASSERT_EQ(h_outIdx[0], params.largeWeightIndex);
  }
}
INSTANTIATE_TEST_CASE_P(SWoRTests, SWoRTestD, ::testing::ValuesIn(inputsd));

}  // end namespace Random
}  // end namespace MLCommon

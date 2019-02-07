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
#include "linalg/map_then_reduce.h"
#include "random/rng.h"
#include "test_utils.h"


namespace MLCommon {
namespace LinAlg {

template <typename Type, typename MapOp>
__global__ void naiveMapReduceKernel(Type *out, const Type *in, size_t len,
                                     MapOp map) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) {
    myAtomicAdd(out, map(in[idx]));
  }
}

template <typename Type, typename MapOp>
void naiveMapReduce(Type *out, const Type *in, size_t len, MapOp map) {
  static const int TPB = 64;
  int nblks = ceildiv(len, (size_t)TPB);
  naiveMapReduceKernel<Type, MapOp><<<nblks, TPB>>>(out, in, len, map);
  CUDA_CHECK(cudaPeekAtLastError());
}


template <typename T>
struct MapReduceInputs {
  T tolerance;
  size_t len;
  unsigned long long int seed;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os, const MapReduceInputs<T> &dims) {
  return os;
}

// Or else, we get the following compilation error
// for an extended __device__ lambda cannot have private or protected access
// within its class
template <typename T>
void mapReduceLaunch(T *out_ref, T *out, const T *in, size_t len) {
  auto op = [] __device__(T in) { return in; };
  naiveMapReduce(out_ref, in, len, op);
  mapThenSumReduce(out, len, op, 0, in);
}

template <typename T>
class MapReduceTest : public ::testing::TestWithParam<MapReduceInputs<T>> {
protected:
  void SetUp() override {
    params = ::testing::TestWithParam<MapReduceInputs<T>>::GetParam();
    Random::Rng<T> r(params.seed);
    auto len = params.len;
    allocate(in, len);
    allocate(out_ref, len);
    allocate(out, len);
    r.uniform(in, len, T(-1.0), T(1.0));
    mapReduceLaunch(out_ref, out, in, len);
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(in));
    CUDA_CHECK(cudaFree(out_ref));
    CUDA_CHECK(cudaFree(out));
  }

protected:
  MapReduceInputs<T> params;
  T *in, *out_ref, *out;
};


const std::vector<MapReduceInputs<float>> inputsf = {
  {0.001f, 1024 * 1024, 1234ULL}};
typedef MapReduceTest<float> MapReduceTestF;
TEST_P(MapReduceTestF, Result) {
  ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                          CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(MapReduceTests, MapReduceTestF,
                        ::testing::ValuesIn(inputsf));


const std::vector<MapReduceInputs<double>> inputsd = {
  {0.000001, 1024 * 1024, 1234ULL}};
typedef MapReduceTest<double> MapReduceTestD;
TEST_P(MapReduceTestD, Result) {
  ASSERT_TRUE(devArrMatch(out_ref, out, params.len,
                          CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(MapReduceTests, MapReduceTestD,
                        ::testing::ValuesIn(inputsd));

} // end namespace LinAlg
} // end namespace MLCommon

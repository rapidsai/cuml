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
#include "stats/weighted_mean.h"
#include "random/rng.h"
#include "test_utils.h"
#include <thrust/device_vector.h>

namespace MLCommon {
namespace Stats {

template <typename T>
struct WeightedMeanInputs {
  T tolerance;
  int M, N;
  unsigned long long int seed;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os, const WeightedMeanInputs<T> &I) {
  return os << "{ " << I.tolerance << ", " << I.M << ", " << I.N << ", "
    << I.seed << "}" << std::endl;
}


///// weighted row-wise mean test and support functions
template <typename T>
void naiveRowWeightedMean(T* R, T* D, T* W, int M, int N, bool rowMajor){
  int istr = rowMajor ? 1 : M;
  int jstr = rowMajor ? N : 1;

  for(int j=0; j<M; j++){
    R[j] = (T)0;
    for(int i=0; i<N; i++){
      R[j] += (W[i]*D[i*istr + j*jstr] - R[j])/(T)(i+1);
    }
  }
}

template <typename T>
class RowWeightedMeanTest : public ::testing::TestWithParam<WeightedMeanInputs<T>> {
protected:
  void SetUp() override {
    params = ::testing::TestWithParam<WeightedMeanInputs<T>>::GetParam();
    Random::Rng r(params.seed);
    int rows = params.M, cols = params.N, len = rows*cols;

    //device-side data
    din.resize(len);
    dweights.resize(cols);
    dexp.resize(rows);
    dact.resize(rows);

    //create random matrix and weights
    r.uniform(din.data().get(), len, T(-1.0), T(1.0));
    r.uniform(dweights.data().get(), cols, T(-1.0), T(1.0));
    
    //host-side data
    thrust::host_vector<T> hin = din;
    thrust::host_vector<T> hweights = dweights;
    thrust::host_vector<T> hexp (rows);

    //compute naive result & copy to GPU
    naiveRowWeightedMean(hexp.data(), hin.data(), hweights.data(), rows, cols,
        true);
    dexp = hexp;

    //compute ml-prims result
    rowWeightedMean(dact.data().get(), din.data().get(), dweights.data().get(),
        cols, rows);

    //adjust tolerance to account for round-off accumulation
    params.tolerance *= params.N;
  }

  void TearDown() override {}

protected:
  WeightedMeanInputs<T> params;
  thrust::host_vector<T> hin, hweights;
  thrust::device_vector<T> din, dweights, dexp, dact;
};


///// weighted column-wise mean test and support functions
template <typename T>
void naiveColWeightedMean(T* R, T* D, T* W, int M, int N, bool rowMajor){
  int istr = rowMajor ? 1 : M;
  int jstr = rowMajor ? N : 1;

  for(int i=0; i<N; i++){
    R[i] = (T)0;
    for(int j=0; j<M; j++){
      R[i] += (W[j]*D[i*istr + j*jstr] - R[i])/(T)(j+1);
    }
  }
}

template <typename T>
class ColWeightedMeanTest : public ::testing::TestWithParam<WeightedMeanInputs<T>> {
  void SetUp() override {
    params = ::testing::TestWithParam<WeightedMeanInputs<T>>::GetParam();
    Random::Rng r(params.seed);
    int rows = params.M, cols = params.N, len = rows*cols;

    //device-side data
    din.resize(len);
    dweights.resize(rows);
    dexp.resize(cols);
    dact.resize(cols);

    //create random matrix and weights
    r.uniform(din.data().get(), len, T(-1.0), T(1.0));
    r.uniform(dweights.data().get(), rows, T(-1.0), T(1.0));
    
    //host-side data
    thrust::host_vector<T> hin = din;
    thrust::host_vector<T> hweights = dweights;
    thrust::host_vector<T> hexp (cols);

    //compute naive result & copy to GPU
    naiveColWeightedMean(hexp.data(), hin.data(), hweights.data(), rows, cols,
        true);
    dexp = hexp;

    //compute ml-prims result
    colWeightedMean(dact.data().get(), din.data().get(), dweights.data().get(),
        cols, rows);

    //adjust tolerance to account for round-off accumulation
    params.tolerance *= params.M;
  }

  void TearDown() override {}

protected:
  WeightedMeanInputs<T> params;
  thrust::host_vector<T> hin, hweights;
  thrust::device_vector<T> din, dweights, dexp, dact;
};


////// Parameter sets and test instantiation
static const float tolF = 128*std::numeric_limits<float>::epsilon();
static const double tolD = 128*std::numeric_limits<double>::epsilon();

const std::vector<WeightedMeanInputs<float>> inputsf = {
  {tolF, 4,  4, 1234},
  {tolF, 1024,  32, 1234},
  {tolF, 1024,  64, 1234},
  {tolF, 1024, 128, 1234},
  {tolF, 1024, 256, 1234},
  {tolF, 1024,  32, 1234},
  {tolF, 1024,  64, 1234},
  {tolF, 1024, 128, 1234},
  {tolF, 1024, 256, 1234}
};

const std::vector<WeightedMeanInputs<double>> inputsd = {
  {tolD, 1024,  32, 1234},
  {tolD, 1024,  64, 1234},
  {tolD, 1024, 128, 1234},
  {tolD, 1024, 256, 1234},
  {tolD, 1024,  32, 1234},
  {tolD, 1024,  64, 1234},
  {tolD, 1024, 128, 1234},
  {tolD, 1024, 256, 1234}
};


using RowWeightedMeanTestF = RowWeightedMeanTest<float>;
TEST_P(RowWeightedMeanTestF, Result){
  ASSERT_TRUE(devArrMatch(dexp.data().get(), dact.data().get(), params.M,
        CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(RowWeightedMeanTest, RowWeightedMeanTestF,
    ::testing::ValuesIn(inputsf));

using RowWeightedMeanTestD = RowWeightedMeanTest<double>;
TEST_P(RowWeightedMeanTestD, Result){
  ASSERT_TRUE(devArrMatch(dexp.data().get(), dact.data().get(), params.M,
        CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(RowWeightedMeanTest, RowWeightedMeanTestD,
    ::testing::ValuesIn(inputsd));


using ColWeightedMeanTestF = ColWeightedMeanTest<float>;
TEST_P(ColWeightedMeanTestF, Result){
  ASSERT_TRUE(devArrMatch(dexp.data().get(), dact.data().get(), params.N,
        CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(ColWeightedMeanTest, ColWeightedMeanTestF,
    ::testing::ValuesIn(inputsf));

using ColWeightedMeanTestD = ColWeightedMeanTest<double>;
TEST_P(ColWeightedMeanTestD, Result){
  ASSERT_TRUE(devArrMatch(dexp.data().get(), dact.data().get(), params.N,
        CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(ColWeightedMeanTest, ColWeightedMeanTestD,
    ::testing::ValuesIn(inputsd));

}; // end namespace Stats
}; // end namespace MLCommon

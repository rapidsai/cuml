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

#include "distance_base.h"

namespace MLCommon {
namespace Distance {

template <typename DataType>
class DistanceUnexpL1 : public DistanceTest<EucUnexpandedL1, DataType> {};

const std::vector<DistanceInputs<float>> inputsf = {
  {0.001f, 1024, 1024, 32, 1234ULL},
  {0.001f, 1024, 32, 1024, 1234ULL},
  {0.001f, 32, 1024, 1024, 1234ULL},
  {0.003f, 1024, 1024, 1024, 1234ULL},
};
typedef DistanceUnexpL1<float> DistanceUnexpL1F;
TEST_P(DistanceUnexpL1F, Result) {
  ASSERT_TRUE(devArrMatch(dist_ref, dist, params.m, params.n,
                          CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(DistanceTests, DistanceUnexpL1F,
                        ::testing::ValuesIn(inputsf));


const std::vector<DistanceInputs<double>> inputsd = {
  {0.001, 1024, 1024, 32, 1234ULL},
  {0.001, 1024, 32, 1024, 1234ULL},
  {0.001, 32, 1024, 1024, 1234ULL},
  {0.003, 1024, 1024, 1024, 1234ULL},
};
typedef DistanceUnexpL1<double> DistanceUnexpL1D;
TEST_P(DistanceUnexpL1D, Result) {
  ASSERT_TRUE(devArrMatch(dist_ref, dist, params.m, params.n,
                          CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(DistanceTests, DistanceUnexpL1D,
                        ::testing::ValuesIn(inputsd));

} // end namespace Distance
} // end namespace MLCommon

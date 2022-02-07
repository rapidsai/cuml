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
#include <iostream>
#include <metrics/kl_divergence.cuh>
#include <raft/cudart_utils.h>
#include <random>

namespace MLCommon {
namespace Metrics {

// parameter structure definition
struct klDivergenceParam {
  int nElements;
  double tolerance;
};

// test fixture class
template <typename DataT>
class klDivergenceTest : public ::testing::TestWithParam<klDivergenceParam> {
 protected:
  // the constructor
  void SetUp() override
  {
    // getting the parameters
    params = ::testing::TestWithParam<klDivergenceParam>::GetParam();

    nElements = params.nElements;

    // generating random value test input
    std::vector<DataT> h_modelPDF(nElements, 0);
    std::vector<DataT> h_candidatePDF(nElements, 0);
    std::random_device rd;
    std::default_random_engine dre(rd());
    std::uniform_real_distribution<DataT> realGenerator(0.0, 1.0);

    std::generate(h_modelPDF.begin(), h_modelPDF.end(), [&]() { return realGenerator(dre); });
    std::generate(
      h_candidatePDF.begin(), h_candidatePDF.end(), [&]() { return realGenerator(dre); });

    // allocating and initializing memory to the GPU
    RAFT_CUDA_TRY(cudaStreamCreate(&stream));

    rmm::device_uvector<DataT> d_modelPDF(nElements, stream);
    rmm::device_uvector<DataT> d_candidatePDF(nElements, stream);
    RAFT_CUDA_TRY(cudaMemset(d_modelPDF.data(), 0, d_modelPDF.size() * sizeof(DataT)));
    RAFT_CUDA_TRY(cudaMemset(d_candidatePDF.data(), 0, d_candidatePDF.size() * sizeof(DataT)));

    raft::update_device(d_modelPDF.data(), &h_modelPDF[0], (int)nElements, stream);
    raft::update_device(d_candidatePDF.data(), &h_candidatePDF[0], (int)nElements, stream);

    // generating the golden output
    for (int i = 0; i < nElements; ++i) {
      if (h_modelPDF[i] == 0.0)
        truthklDivergence += 0;

      else
        truthklDivergence += h_modelPDF[i] * log(h_modelPDF[i] / h_candidatePDF[i]);
    }

    // calling the kl_divergence CUDA implementation
    computedklDivergence =
      MLCommon::Metrics::kl_divergence(d_modelPDF.data(), d_candidatePDF.data(), nElements, stream);
    RAFT_CUDA_TRY(cudaStreamDestroy(stream));
  }

  // declaring the data values
  klDivergenceParam params;
  int nElements              = 0;
  DataT truthklDivergence    = 0;
  DataT computedklDivergence = 0;
  cudaStream_t stream        = 0;
};

// setting test parameter values
const std::vector<klDivergenceParam> inputs = {
  {500, 0.000001}, {200, 0.001}, {5000, 0.000001}, {500000, 0.000001}

};

// writing the test suite
typedef klDivergenceTest<double> klDivergenceTestClass;
TEST_P(klDivergenceTestClass, Result)
{
  ASSERT_NEAR(computedklDivergence, truthklDivergence, params.tolerance);
}
INSTANTIATE_TEST_CASE_P(klDivergence, klDivergenceTestClass, ::testing::ValuesIn(inputs));

}  // end namespace Metrics
}  // end namespace MLCommon

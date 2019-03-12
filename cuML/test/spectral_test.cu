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

#include "spectral/spectral.h"

#include "random/rng.h"

#include <vector>
#include <gtest/gtest.h>
#include <cuda_utils.h>
#include <test_utils.h>
#include <iostream>

namespace ML {

using namespace MLCommon;

template<typename T>
class SpectralTest: public ::testing::Test {
protected:
    void SetUp() override {}

    void TearDown() override {}

protected:

};


typedef SpectralTest<float> TestSpectralClustering;
TEST_F(TestSpectralClustering, Fit) {

    int n = 500;
    int d = 30;
    int k = 3;

    float *X;
    MLCommon::allocate(X, n*d);

    Random::Rng r(150, MLCommon::Random::GenTaps);
    r.uniform(X, n*d, -1.0f, 1.0f);

    int *out;
    MLCommon::allocate(out, n, true);

    ML::Spectral::fit_clusters(X, n, d, k, 10, 1e-3f, out);

    std::cout << MLCommon::arr2Str(out, n, "out") << std::endl;
}

typedef SpectralTest<float> TestSpectralEmbedding;
TEST_F(TestSpectralEmbedding, Fit) {

    int n = 500;
    int d = 30;
    int k = 3;

    float *X;
    MLCommon::allocate(X, n*d);

    Random::Rng r(150, MLCommon::Random::GenTaps);
    r.uniform(X, n*d, -1.0f, 1.0f);

    float *out;
    MLCommon::allocate(out, n*2, true);

    ML::Spectral::fit_embedding(X, n, d, k, 2, out);

    std::cout << MLCommon::arr2Str(out, n, "out") << std::endl;
}


} // end namespace ML

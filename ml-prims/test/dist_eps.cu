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

#include "distance/distance.h"

#include <cuda_utils.h>
#include "test_utils.h"

#include <vector>
#include <iostream>


namespace MLCommon {
namespace Distance {


class EpsilonNeighborhoodTest: public ::testing::Test {

protected:
    void SetUp() override {}
    void TearDown() override {}
};


typedef EpsilonNeighborhoodTest TestNeighborhoodsNoFunctor;
TEST_F(TestNeighborhoodsNoFunctor, Result) {

    cudaStream_t stream;
    CUDA_CHECK( cudaStreamCreate(&stream) );

    int m = 6;
    int k = 2;

    float *data;
    bool *adj, *expected;

    allocate(data, m*k, true);
    allocate(adj, m*m, true);
    allocate(expected, m*m, true);

    std::vector<float> data_h = { 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 8.0, 7.0, 8.0, 8.0, 25.0, 80.0 };
    data_h.resize(m*k);

    bool *expected_h = new bool[m*m]{
            1,    1,    1,    0,    0,    0,
            1,    1,    1,    0,    0,    0,
            1,    1,    1,    0,    0,    0,
            0,    0,    0,    1,    1,    0,
            0,    0,    0,    1,    1,    0,
            0,    0,    0,    0,    0,    1
    };

    updateDevice(data, data_h.data(), m*k, stream);
    updateDevice(expected, expected_h, m*m, stream);

    float eps = 3.0;

    char* workspace;
    size_t workspaceSize = 0;

    constexpr auto distance_type = MLCommon::Distance::DistanceType::EucExpandedL2Sqrt;

    workspaceSize =  MLCommon::Distance::getWorkspaceSize
            <distance_type, float, float, bool>(data, data, m, m, k);

    if (workspaceSize != 0)
        allocate(workspace, workspaceSize, true);

    epsilon_neighborhood<distance_type, float>
        (data, data, adj, m, m, k, eps, (void*)workspace, workspaceSize, stream);

    CUDA_CHECK( cudaStreamSynchronize(stream) );

    ASSERT_TRUE(devArrMatch(adj, expected, m*m, Compare<bool>(), stream));

    CUDA_CHECK( cudaStreamDestroy(stream) );
    CUDA_CHECK( cudaFree(data) );
    CUDA_CHECK( cudaFree(adj) );

    delete expected_h;
}
};
};

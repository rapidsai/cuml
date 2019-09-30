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
#include <score/scores.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include "datasets/digits.h"
#include "cuda_utils.h"
#include "cuML.hpp"

#include "common/cuml_allocator.hpp"
#include "common/device_buffer.hpp"

using namespace std;
using namespace MLCommon;
using namespace MLCommon::Score;
using namespace MLCommon::Distance;
using namespace MLCommon::Datasets::Digits;
using namespace ML;



template <DistanceType distance_type>
static void
get_distances(float *X,
              float *output_D,
              int n,
              int p,
              std::shared_ptr<deviceAllocator> d_alloc,
              cudaStream_t stream)
{
  typedef cutlass::Shape<8, 128, 128> OutputTile_t;

  // Determine distance workspace size
  char *distance_workspace = nullptr;
  const size_t distance_workspace_size = \
    MLCommon::Distance::getWorkspaceSize<distance_type, float, float, float>(
      X, X, n, n, p);

  printf("Workspace = %zu\n", distance_workspace_size);
  
  if (distance_workspace_size != 0)
    distance_workspace = (char*) d_alloc->allocate(distance_workspace_size, stream);
  

  // Find distances
  MLCommon::Distance::distance<distance_type, float, float, float, OutputTile_t>(
    X, X, output_D, n, n, p,
    (void*)distance_workspace, distance_workspace_size, stream);

  CUDA_CHECK(cudaPeekAtLastError());

  
  // Free workspace
  if (distance_workspace_size != 0)
    d_alloc->deallocate(distance_workspace, distance_workspace_size, stream);
}




class DistanceTest : public ::testing::Test
{
 protected:
  void basicTest()
  {
    cumlHandle handle;
    auto d_alloc = handle.getDeviceAllocator();
    auto stream = handle.getStream();

    // Allocate memory
    device_buffer<float> X_d(d_alloc, stream, n*p);
    MLCommon::updateDevice(X_d.data(), digits.data(), n*p, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    device_buffer<float> output_D(d_alloc, stream, n*n);

    // Test each distance type
    printf("Testing distance type = %d\n", EucExpandedL2);
    get_distances<EucExpandedL2>(X_d.data(), output_D.data(), n, p, d_alloc, stream);

    printf("Testing distance type = %d\n", EucExpandedL2Sqrt);
    get_distances<EucExpandedL2Sqrt>(X_d.data(), output_D.data(), n, p, d_alloc, stream);

    printf("Testing distance type = %d\n", EucExpandedCosine);
    get_distances<EucExpandedCosine>(X_d.data(), output_D.data(), n, p, d_alloc, stream);

    printf("Testing distance type = %d\n", EucUnexpandedL1);
    get_distances<EucUnexpandedL1>(X_d.data(), output_D.data(), n, p, d_alloc, stream);

    printf("Testing distance type = %d\n", EucUnexpandedL2);
    get_distances<EucUnexpandedL2>(X_d.data(), output_D.data(), n, p, d_alloc, stream);

    printf("Testing distance type = %d\n", EucUnexpandedL2Sqrt);
    get_distances<EucUnexpandedL2Sqrt>(X_d.data(), output_D.data(), n, p, d_alloc, stream);
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  int n = 1797;
  int p = 64;
};

typedef DistanceTest DistanceTestF;
TEST_F(DistanceTestF, Result)
{
}

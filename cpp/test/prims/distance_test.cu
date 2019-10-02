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
#include <iostream>
#include <vector>
#include "datasets/digits.h"
#include <cuda_utils.h>
#include "distance/distance.h"

#include "common/cuml_allocator.hpp"
#include "common/device_buffer.hpp"

using namespace std;
using namespace MLCommon;
using namespace MLCommon::Score;
using namespace MLCommon::Distance;
using namespace MLCommon::Datasets::Digits;



template <DistanceType d_type, typename T>
static void
get_distances(T *X,
              T *output_D,
              int n,
              int p,
              std::shared_ptr<deviceAllocator> d_alloc,
              cudaStream_t stream)
{
  typedef cutlass::Shape<8, 128, 128> OutputTile_t;

  // Determine distance workspace size
  const size_t lwork = getWorkspaceSize<d_type, T, T, T>(X, X, n, n, p);
  void *work = (lwork > 0) ? ((void*) d_alloc->allocate(lwork, stream)) : NULL;
  
  // Find distances
  MLCommon::Distance::distance<d_type, T, T, T, OutputTile_t>(
    X, X, output_D, n, n, p, work, lwork, stream);
  CUDA_CHECK(cudaPeekAtLastError());
  
  // Free workspace
  if (lwork > 0) d_alloc->deallocate(work, lwork, stream);
}



template <DistanceType d_type>
class DistanceTest : public ::testing::Test
{
 protected:
  void basicTest()
  {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    d_alloc.reset(new defaultDeviceAllocator);

    // Allocate memory
    device_buffer<float> X_d(d_alloc, stream, n*p);
    MLCommon::updateDevice(X_d.data(), digits.data(), n*p, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    device_buffer<float> output_D(d_alloc, stream, n*n);

    // Test each distance type
    get_distances<d_type>(X_d.data(), output_D.data(), n, p, d_alloc, stream);

    cudaStreamDestroy(stream);
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  int n = 1797;
  int p = 64;
  std::shared_ptr<deviceAllocator> d_alloc;
};


typedef DistanceTest<EucExpandedL2> DistanceTest_EucExpandedL2;
TEST_F(DistanceTest_EucExpandedL2, Result) {}

typedef DistanceTest<EucExpandedL2Sqrt> DistanceTest_EucExpandedL2Sqrt;
TEST_F(DistanceTest_EucExpandedL2Sqrt, Result) {}

typedef DistanceTest<EucExpandedCosine> DistanceTest_EucExpandedCosine;
TEST_F(DistanceTest_EucExpandedCosine, Result) {}

typedef DistanceTest<EucUnexpandedL1> DistanceTest_EucUnexpandedL1;
TEST_F(DistanceTest_EucUnexpandedL1, Result) {}

typedef DistanceTest<EucUnexpandedL2> DistanceTest_EucUnexpandedL2;
TEST_F(DistanceTest_EucUnexpandedL2, Result) {}

typedef DistanceTest<EucUnexpandedL2Sqrt> DistanceTest_EucUnexpandedL2Sqrt;
TEST_F(DistanceTest_EucUnexpandedL2Sqrt, Result) {}

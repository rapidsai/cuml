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

#include "svm/workingset.h"
#include "svm/smosolver.h"
#include <gtest/gtest.h>
#include <cuda_utils.h>
#include <test_utils.h>
#include <iostream>
#include <cub/cub.cuh>   // or equivalently <cub/device/device_radix_sort.cuh>

namespace ML {
namespace SVM {
using namespace MLCommon;


/*
template<typename math_t>
class SmoSolverTest: public ::testing::Test {
protected:
   // SmoSolver<math_t> * smo;
    SmoSolverTest() 
    {
     // smo = new SmoSolver<math_t>(10,4);
    }
    ~ SmoSolverTest() {
      //delete smo;
    }
};

typedef SmoSolverTest<float> SmoSolverTestF;
*/


TEST(SmoSolverTestF, SelectWorkingSetTest) {
  WorkingSet<float> *ws;
  
  ws = new WorkingSet<float>(10);
  EXPECT_EQ(ws->GetSize(), 10);
  delete ws;
  
  ws = new WorkingSet<float>(100000);
  EXPECT_EQ(ws->GetSize(), 1024);
  delete ws;

  ws = new WorkingSet<float>(10, 4);
  EXPECT_EQ(ws->GetSize(), 4);
  
  float f_host[10] = {1, 3, 10, 4, 2, 8, 6, 5, 9, 7};
  float *f_dev;

  float y_host[10] = {-1, -1, -1, -1, -1, 1, 1, 1, 1, 1};
  float *y_dev;
  
  float C=1.5;
  
  float alpha_host[10] = {0, 0, 0.1, 0.2, 1.5, 0, 0.2, 0.4, 1.5, 1.5 };
  float *alpha_dev;  //   l  l  l/u  l/u    u  u  l/u  l/u  l    l
    
  int expected_idx[4] = {4, 7, 2, 8};
  //int expected_idx[4] = {2, 5, 0, 1};
  allocate(f_dev, 10);
  allocate(y_dev, 10);
  allocate(alpha_dev, 10);
  updateDevice(f_dev, f_host, 10);
  updateDevice(y_dev, y_host, 10); 
  updateDevice(alpha_dev, alpha_host, 10);
  
  ws->Select(f_dev, alpha_dev, y_dev, C);
  int idx[4];
  updateHost(idx, ws->idx, 4);  
  for (int i=0; i<4; i++) {
    EXPECT_EQ(idx[i], expected_idx[i]);
  }
  CUDA_CHECK(cudaFree(f_dev));
  CUDA_CHECK(cudaFree(y_dev));
  CUDA_CHECK(cudaFree(alpha_dev));
  delete ws;
}

/*TEST_F(SmoSolverTestF, SelectWorkingSetTest) {
  ASSERT_LT(1, 2);
}*/

}; // end namespace SVM
}; // end namespace ML

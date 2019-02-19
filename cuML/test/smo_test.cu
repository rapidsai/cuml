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

#include "svm/smo.h"
#include <gtest/gtest.h>
#include <cuda_utils.h>
#include <test_utils.h>
#include <iostream>

namespace ML {
namespace SVM {
using namespace MLCommon;


/**
 *
 * NOTE: Not exhaustively testing the kNN implementation since
 * we are using FAISS for this. Just testing API to verify the
 * knn.cu class is accepting inputs and providing outputs as
 * expected.
 */
template<typename math_t>
class SmoSolverTest: public ::testing::Test {
protected:
    SmoSolver<math_t> * smo;
    SmoSolverTest() 
    {
      smo = new SmoSolver<math_t>(10,4);
    }
    ~ SmoSolverTest() {
      delete smo;
    }
};

typedef SmoSolverTest<float> SmoSolverTestF;

TEST_F(SmoSolverTestF, SelectWorkingSetTest) {
  ASSERT_LT(1, 2);
}

}; // end namespace SVM
}; // end namespace ML

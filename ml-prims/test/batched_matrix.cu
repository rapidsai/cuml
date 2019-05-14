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
#include "matrix/batched_matrix.h"
#include "random/rng.h"
#include "test_utils.h"

namespace MLCommon {
namespace Matrix {

template <typename T>
struct BatchedMatrixInputs {
  T tolerance;
};

template <typename T>
class BatchedMatrixTest : public ::testing::TestWithParam<BatchedMatrixInputs<T>> {
protected:
  void SetUp() override {
    using std::vector;
    params = ::testing::TestWithParam<BatchedMatrixInputs<T>>::GetParam();

    //////////////////////////////////////////////////////////////
    // Reference matrices
    // NOTE: cublas expects in column major.
    // 2x2
    std::vector<T> A = {0.22814838,
                        0.92204276,
                        /*       */0.32118359,
                        /*       */0.28488466};
    // A = np.array([[0.22814838,0.32118359],[0.92204276,0.28488466]])

    // 2x2
    std::vector<T> B = {0.1741319,
                        0.19051178,
                        /*       */0.21628607,
                        /*       */0.35775104
    };
    // B = np.array([[0.1741319,0.21628607],[0.19051178,0.35775104]])

    // 1x2
    std::vector<T> Z = {0.11387309, 0.21870136};

    //////////////////////////////////////////////////////////////
    // reference via numpy
    // A@B = array([[0.10091717, 0.16424908],
    //              [0.21483094, 0.30134279]])
    vector<T> ABref =
      {0.10091717,
       0.21483094,
       /*       */0.16424908,
       /*       */0.30134279};
    T* d_ABref;
    allocate(d_ABref, 4);
    updateDevice(d_ABref, ABref.data(), ABref.size(), 0);

    // B@Z.T = array([[0.067131 ],
    //                [0.0999348]])
    vector<T> BZTref = {0.067131,
                        0.0999348};
    T* d_BZTref;
    allocate(d_BZTref, 2);
    updateDevice(d_BZTref, BZTref.data(), BZTref.size(), 0);

    // Z@B = array([[0.06149412, 0.1028698 ]])
    vector<T> ZBref = {0.06149412, 0.1028698};
    T* d_ZBref;
    allocate(d_ZBref, 2);
    updateDevice(d_ZBref, ZBref.data(), ZBref.size(), 0);

    vector<T> ApBref = {
                        0.40228028,
                        1.11255454,
                        /*       */ 0.53746966,
                        /*       */ 0.6426357
    };
    T* d_ApBref;
    allocate(d_ApBref, 4);
    updateDevice(d_ApBref, ApBref.data(), ApBref.size(), 0);

    vector<T> AmBref = {
                        0.05401648,
                        0.73153098,
                        /*       */ 0.10489752,
                        /*       */ -0.07286638
    };
    T* d_AmBref;
    allocate(d_AmBref, 4);
    updateDevice(d_AmBref, AmBref.data(), AmBref.size(), 0);

    // A+B = array([[0.40228028, 0.53746966],[1.11255454, 0.6426357 ]])
    // A-B = array([[ 0.05401648,  0.10489752],[ 0.73153098, -0.07286638]])

    //////////////////////////////////////////////////////////////
    // setup gpu memory
    int num_batches = 3;
    // T* Abi;
    // allocate(Abi, 4*num_batches);
    // T* Bbi;
    // allocate(Bbi, 4*num_batches);
    // T* Zbi;
    // allocate(Zbi, 2*num_batches);

    auto memory_pool = std::make_shared<BatchedMatrixMemoryPool>(num_batches);

    BatchedMatrix AbM(2, 2, num_batches, memory_pool);
    BatchedMatrix BbM(2, 2, num_batches, memory_pool);
    BatchedMatrix ZbM(1, 2, num_batches, memory_pool);
    for(int i=0;i<num_batches;i++) {
      updateDevice(AbM[i], A.data(), 4, 0);
      updateDevice(BbM[i], B.data(), 4, 0);
      updateDevice(ZbM[i], Z.data(), 2, 0);
    }

    //////////////////////////////////////////////////////////////
    // compute
    std::cout << "AB\n";
    BatchedMatrix AB = AbM*BbM;
    std::cout << "ZB\n";
    BatchedMatrix ZB = ZbM*BbM;
    std::cout << "BZT\n";
    BatchedMatrix BZT = b_gemm(BbM,ZbM, false, true);

    std::cout << "A+B\n";
    BatchedMatrix A_p_B = AbM + BbM;
    std::cout << "A-B\n";
    BatchedMatrix A_m_B = AbM - BbM;

    std::cout << "AB * B + B*AB - AB + B - A\n";
    BatchedMatrix TestAlloc = AB * BbM + BbM*AB - AB + BbM - AbM;
    std::cout << "2:AB * B + B*AB - AB + B - A\n";
    BatchedMatrix TestAlloc2 = AB * BbM + BbM*AB - AB + BbM - AbM;
    std::cout << "3:AB * B\n";
    BatchedMatrix TestAlloc3 = AB * BbM;

    //////////////////////////////////////////////////////////////
    // compare answers
    for(int i=0;i<num_batches;i++) {
      ASSERT_TRUE(devArrMatch(d_ABref, AB[i], AB.shape().first, AB.shape().second,
                              CompareApprox<T>(params.tolerance)));
      ASSERT_TRUE(devArrMatch(d_ZBref, ZB[i], ZB.shape().first, ZB.shape().second,
                              CompareApprox<T>(params.tolerance)));
      ASSERT_TRUE(devArrMatch(d_BZTref, BZT[i], BZT.shape().first, BZT.shape().second,
                              CompareApprox<T>(params.tolerance)));
      ASSERT_TRUE(devArrMatch(d_ApBref, A_p_B[i], A_p_B.shape().first, A_p_B.shape().second,
                              CompareApprox<T>(params.tolerance)));
      ASSERT_TRUE(devArrMatch(d_AmBref, A_m_B[i], A_m_B.shape().first, A_m_B.shape().second,
                              CompareApprox<T>(params.tolerance)));

      // Compare two different matrices to check failure case
      ASSERT_FALSE(devArrMatch(d_ABref, A_m_B[i], A_m_B.shape().first, A_m_B.shape().second,
                               CompareApprox<T>(params.tolerance)));
    }
  }

  void TearDown() override {
  }

protected:
  BatchedMatrixInputs<T> params;
};

using BatchedMatrixTestD = BatchedMatrixTest<double>;
TEST_P(BatchedMatrixTestD, Result) {
  std::cout << "Finished Test\n";
}

const std::vector<BatchedMatrixInputs<double>> inputsd = {{1e-6}};

INSTANTIATE_TEST_CASE_P(BatchedMatrixTests, BatchedMatrixTestD,
                        ::testing::ValuesIn(inputsd));

}
}

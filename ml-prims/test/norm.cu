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
#include "linalg/norm.h"
#include "random/rng.h"
#include "test_utils.h"

namespace MLCommon {
namespace LinAlg {

template <typename T>
struct NormInputs {
  T tolerance;
  int rows, cols;
  NormType type;
  bool do_sqrt;
  unsigned long long int seed;
};

template <typename T>
::std::ostream &operator<<(::std::ostream &os, const NormInputs<T> &I) {
  os << "{ " << I.tolerance << ", " << I.rows << ", " << I.cols << ", "
    << I.type << ", " << I.do_sqrt << ", "
    << I.seed << '}' << std::endl;
  return os;
}


///// Row-wise norm test definitions
template <typename Type>
__global__ void naiveRowNormKernel(Type *dots, const Type *data, int D, int N,
                                NormType type, bool do_sqrt) {
  Type acc = (Type)0;
  int rowStart = threadIdx.x + blockIdx.x * blockDim.x;
  if (rowStart < N) {
    for (int i = 0; i < D; ++i) {
      if (type == L2Norm) {
        acc += data[rowStart * D + i] * data[rowStart * D + i];
      } else {
        acc += myAbs(data[rowStart * D + i]);
      }
    }
    dots[rowStart] = do_sqrt ? mySqrt(acc) : acc;
  }
}

template <typename Type>
void naiveRowNorm(Type *dots, const Type *data, int D, int N, NormType type,
               bool do_sqrt, cudaStream_t stream) {
  static const int TPB = 64;
  int nblks = ceildiv(N, TPB);
  naiveRowNormKernel<Type><<<nblks, TPB, 0, stream>>>(dots, data, D, N, type, do_sqrt);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
class RowNormTest : public ::testing::TestWithParam<NormInputs<T>> {
public:
  void SetUp() override {
    params = ::testing::TestWithParam<NormInputs<T>>::GetParam();
    Random::Rng r(params.seed);
    int rows = params.rows, cols = params.cols, len = rows * cols;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    allocate(data, len);
    allocate(dots_exp, rows);
    allocate(dots_act, rows);
    r.uniform(data, len, T(-1.0), T(1.0), stream);
    naiveRowNorm(dots_exp, data, cols, rows, params.type, params.do_sqrt, stream);
    if (params.do_sqrt) {
      auto fin_op = [] __device__(T in) { return mySqrt(in); };
      rowNorm(dots_act, data, cols, rows, params.type, fin_op, stream);
    } else {
      rowNorm(dots_act, data, cols, rows, params.type, stream);
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(dots_exp));
    CUDA_CHECK(cudaFree(dots_act));
  }

protected:
  NormInputs<T> params;
  T *data, *dots_exp, *dots_act;
};


///// Column-wise norm test definitisons
template <typename Type>
__global__ void naiveColNormKernel(Type *dots, const Type *data, int D, int N,
                                   NormType type, bool do_sqrt) {
    int colID = threadIdx.x + blockIdx.x*blockDim.x;
    if(colID > D) return; //avoid out-of-bounds thread

    Type acc = 0;
    for(int i=0; i<N; i++){
      Type v = data[colID + i*D];
      acc += type==L2Norm ? v*v : myAbs(v);
    }

    dots[colID] = do_sqrt ? mySqrt(acc) : acc;
}

template <typename Type>
void naiveColNorm(Type *dots, const Type *data, int D, int N, NormType type,
               bool do_sqrt, cudaStream_t stream) {
  static const int TPB = 64;
  int nblks = ceildiv(D, TPB);
  naiveColNormKernel<Type><<<nblks,TPB, 0, stream>>>(dots, data, D, N, type, do_sqrt);
  CUDA_CHECK(cudaPeekAtLastError());
}


template <typename T>
class ColNormTest : public ::testing::TestWithParam<NormInputs<T>> {
public:
  void SetUp() override {
    params = ::testing::TestWithParam<NormInputs<T>>::GetParam();
    Random::Rng r(params.seed);
    int rows = params.rows, cols = params.cols, len = rows * cols;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    allocate(data, len);
    r.uniform(data, len, T(-1.0), T(1.0), stream);
    allocate(dots_exp, cols);
    allocate(dots_act, cols);

    naiveColNorm(dots_exp, data, cols, rows, params.type, params.do_sqrt, stream);
    if(params.do_sqrt){
      auto fin_op = [] __device__(T in) { return mySqrt(in); };
      colNorm(dots_act, data, cols, rows, params.type, fin_op, stream);
    }else{
      colNorm(dots_act, data, cols, rows, params.type, stream);
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void TearDown() override {
    CUDA_CHECK(cudaFree(data));
    CUDA_CHECK(cudaFree(dots_exp));
    CUDA_CHECK(cudaFree(dots_act));
  }

protected:
    NormInputs<T> params;
    T *data, *dots_exp, *dots_act;
};



///// Row- and column-wise tests
const std::vector<NormInputs<float>> inputsf = {
  {0.00001f, 1024, 32, L1Norm, false, 1234ULL},
  {0.00001f, 1024, 64, L1Norm, false, 1234ULL},
  {0.00001f, 1024, 128, L1Norm, false, 1234ULL},
  {0.00001f, 1024, 256, L1Norm, false, 1234ULL},
  {0.00001f, 1024, 32, L2Norm, false, 1234ULL},
  {0.00001f, 1024, 64, L2Norm, false, 1234ULL},
  {0.00001f, 1024, 128, L2Norm, false, 1234ULL},
  {0.00001f, 1024, 256, L2Norm, false, 1234ULL},

  {0.00001f, 1024, 32, L1Norm, true, 1234ULL},
  {0.00001f, 1024, 64, L1Norm, true, 1234ULL},
  {0.00001f, 1024, 128, L1Norm, true, 1234ULL},
  {0.00001f, 1024, 256, L1Norm, true, 1234ULL},
  {0.00001f, 1024, 32, L2Norm, true, 1234ULL},
  {0.00001f, 1024, 64, L2Norm, true, 1234ULL},
  {0.00001f, 1024, 128, L2Norm, true, 1234ULL},
  {0.00001f, 1024, 256, L2Norm, true, 1234ULL}
};

const std::vector<NormInputs<double>> inputsd = {
  {0.00000001, 1024, 32, L1Norm, false, 1234ULL},
  {0.00000001, 1024, 64, L1Norm, false, 1234ULL},
  {0.00000001, 1024, 128, L1Norm, false, 1234ULL},
  {0.00000001, 1024, 256, L1Norm, false, 1234ULL},
  {0.00000001, 1024, 32, L2Norm, false, 1234ULL},
  {0.00000001, 1024, 64, L2Norm, false, 1234ULL},
  {0.00000001, 1024, 128, L2Norm, false, 1234ULL},
  {0.00000001, 1024, 256, L2Norm, false, 1234ULL},

  {0.00000001, 1024, 32, L1Norm, true, 1234ULL},
  {0.00000001, 1024, 64, L1Norm, true, 1234ULL},
  {0.00000001, 1024, 128, L1Norm, true, 1234ULL},
  {0.00000001, 1024, 256, L1Norm, true, 1234ULL},
  {0.00000001, 1024, 32, L2Norm, true, 1234ULL},
  {0.00000001, 1024, 64, L2Norm, true, 1234ULL},
  {0.00000001, 1024, 128, L2Norm, true, 1234ULL},
  {0.00000001, 1024, 256, L2Norm, true, 1234ULL}
};

typedef RowNormTest<float> RowNormTestF;
TEST_P(RowNormTestF, Result) {
  ASSERT_TRUE(devArrMatch(dots_exp, dots_act, params.rows,
                          CompareApprox<float>(params.tolerance)));
}

typedef RowNormTest<double> RowNormTestD;
TEST_P(RowNormTestD, Result) {
  ASSERT_TRUE(devArrMatch(dots_exp, dots_act, params.rows,
                          CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(RowNormTests, RowNormTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(RowNormTests, RowNormTestD, ::testing::ValuesIn(inputsd));


const std::vector<NormInputs<float>> inputscf = {
  {0.00001f, 32,  1024, L1Norm, false, 1234ULL},
  {0.00001f, 64,  1024, L1Norm, false, 1234ULL},
  {0.00001f, 128, 1024, L1Norm, false, 1234ULL},
  {0.00001f, 256, 1024, L1Norm, false, 1234ULL},
  {0.00001f, 32,  1024, L2Norm, false, 1234ULL},
  {0.00001f, 64,  1024, L2Norm, false, 1234ULL},
  {0.00001f, 128, 1024, L2Norm, false, 1234ULL},
  {0.00001f, 256, 1024, L2Norm, false, 1234ULL},

  {0.00001f, 32,  1024, L1Norm, true, 1234ULL},
  {0.00001f, 64,  1024, L1Norm, true, 1234ULL},
  {0.00001f, 128, 1024, L1Norm, true, 1234ULL},
  {0.00001f, 256, 1024, L1Norm, true, 1234ULL},
  {0.00001f, 32,  1024, L2Norm, true, 1234ULL},
  {0.00001f, 64,  1024, L2Norm, true, 1234ULL},
  {0.00001f, 128, 1024, L2Norm, true, 1234ULL},
  {0.00001f, 256, 1024, L2Norm, true, 1234ULL}
};

const std::vector<NormInputs<double>> inputscd = {
  {0.00000001, 32,  1024, L1Norm, false, 1234ULL},
  {0.00000001, 64,  1024, L1Norm, false, 1234ULL},
  {0.00000001, 128, 1024, L1Norm, false, 1234ULL},
  {0.00000001, 256, 1024, L1Norm, false, 1234ULL},
  {0.00000001, 32,  1024, L2Norm, false, 1234ULL},
  {0.00000001, 64,  1024, L2Norm, false, 1234ULL},
  {0.00000001, 128, 1024, L2Norm, false, 1234ULL},
  {0.00000001, 256, 1024, L2Norm, false, 1234ULL},

  {0.00000001, 32,  1024, L1Norm, true, 1234ULL},
  {0.00000001, 64,  1024, L1Norm, true, 1234ULL},
  {0.00000001, 128, 1024, L1Norm, true, 1234ULL},
  {0.00000001, 256, 1024, L1Norm, true, 1234ULL},
  {0.00000001, 32,  1024, L2Norm, true, 1234ULL},
  {0.00000001, 64,  1024, L2Norm, true, 1234ULL},
  {0.00000001, 128, 1024, L2Norm, true, 1234ULL},
  {0.00000001, 256, 1024, L2Norm, true, 1234ULL}
};

typedef ColNormTest<float> ColNormTestF;
TEST_P(ColNormTestF, Result) {
  ASSERT_TRUE(devArrMatch(dots_exp, dots_act, params.cols,
                          CompareApprox<float>(params.tolerance)));
}

typedef ColNormTest<double> ColNormTestD;
TEST_P(ColNormTestD, Result) {
  ASSERT_TRUE(devArrMatch(dots_exp, dots_act, params.cols,
                          CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(ColNormTests, ColNormTestF, ::testing::ValuesIn(inputscf));

INSTANTIATE_TEST_CASE_P(ColNormTests, ColNormTestD, ::testing::ValuesIn(inputscd));

} // end namespace LinAlg
} // end namespace MLCommon

#ifndef IF_DEBUG
#define IF_DEBUG 1
#endif

#include <gtest/gtest.h>
#include "tsne/digits.h"
#include "tsne/tsne.cu"
//#include "tsne/Ground_Truth_TSNE.h"
#include <metrics/trustworthiness.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

#include "cuda_utils.h"

using namespace MLCommon;
using namespace ML::Metrics;
using namespace MLCommon::Distance;

using namespace ML;

class TSNETest : public ::testing::Test {
 protected:
  void basicTest() {
    cumlHandle handle;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    float *X_d, *Y_d;
    MLCommon::allocate(X_d, n * p);
    MLCommon::allocate(Y_d, n * 2);
    MLCommon::updateDevice(X_d, digits.data(), n * p, stream);

    std::cout << "[>>>>]    Starting TSNE....\n";
    TSNE(handle, X_d, Y_d, n, p);
    std::cout << "[>>>>]    Got embeddings!....\n";

    std::cout << MLCommon::arr2Str(Y_d, 20, "embeddings", stream) << std::endl;

    std::cout << "Updating host" << std::endl;
    float embeddings_h[n * 2];
    MLCommon::updateHost(embeddings_h, Y_d, n * 2, stream);

    std::cout << "DONE!" << std::endl;

    CUDA_CHECK(cudaPeekAtLastError());

    // Test trustworthiness
    // euclidean test
    score = trustworthiness_score<float, EucUnexpandedL2Sqrt>(handle, X_d, Y_d,
                                                              n, p, 2, 90);

    std::cout << "SCORE: " << score << std::endl;

    CUDA_CHECK(cudaFree(Y_d));
    CUDA_CHECK(cudaFree(X_d));

    CUDA_CHECK(cudaStreamDestroy(stream));
  }

  void SetUp() override { basicTest(); }

  void TearDown() override {}

 protected:
  int n = 1797;
  int p = 64;
  double score;
};

typedef TSNETest TSNETestF;
TEST_F(TSNETestF, Result) { ASSERT_TRUE(0.9374 < score && score < 0.9376); }

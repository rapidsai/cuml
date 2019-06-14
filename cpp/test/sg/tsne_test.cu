#ifndef IF_DEBUG
#define IF_DEBUG 1
#endif

#include <gtest/gtest.h>
#include "tsne/digits.h"
#include "tsne/tsne.cu"
//#include "tsne/Ground_Truth_TSNE.h"
#include <score/scores.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>

#include "cuda_utils.h"

using namespace MLCommon;
using namespace MLCommon::Score;
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
    MLCommon::updateDevice(Y_d, Y.data(), n * 2, stream);

    std::cout << "[>>>>]    Starting TSNE....\n";
    TSNE(handle, X_d, Y_d, n, p, 2, 5, distances.data(), indices.data(), VAL_vector.data(), COL_vector.data(),
    	ROW_vector.data());
    std::cout << "[>>>>]    Got embeddings!....\n";

    std::cout << MLCommon::arr2Str(Y_d, 20, "embeddings", stream) << std::endl;
    std::cout << MLCommon::arr2Str(Y_d+n/2, 20, "embeddings", stream) << std::endl;

    std::cout << MLCommon::arr2Str(Y_d+n, 20, "embeddings", stream) << std::endl;
    std::cout << MLCommon::arr2Str(Y_d+n+n/2, 20, "embeddings", stream) << std::endl;

    std::cout << "Updating host" << std::endl;
    float embeddings_h[n * 2];

    int k = 0;
    for (int i = 0; i < n; i++) {
    	for (int j = 0; j < k; j++) {
    		cudaMemcpy(&embeddings_h[k], Y_d+j*n+i, sizeof(float), cudaMemcpyDeviceToHost);
    	}
    }
    for (int j = 0; j < 2; j++) {
    	for (int i = 0; i < n; i++)
    		printf("%.f", embeddings_h[i*2 + j]);
    	printf("\n\n");
    }
    float *YY; MLCommon::allocate(YY, n * 2);
    MLCommon::updateDevice(YY, embeddings_h, n * 2, stream);
    		
    //MLCommon::updateHost(embeddings_h, Y_d, n * 2, stream);

    std::cout << "DONE!" << std::endl;

    CUDA_CHECK(cudaPeekAtLastError());

    // Test trustworthiness
    // euclidean test
    score = trustworthiness_score<float, EucUnexpandedL2>(
      X_d, YY, n, p, 2, 5, handle.getDeviceAllocator(), stream);

    std::cout << "SCORE: " << score << std::endl;

    CUDA_CHECK(cudaFree(Y_d));
    CUDA_CHECK(cudaFree(YY));
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

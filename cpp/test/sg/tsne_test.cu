#ifndef IF_DEBUG
    #define IF_DEBUG 1 
#endif

#include <gtest/gtest.h>
#include "tsne/tsne.h"
#include "tsne/digits.h"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>

using namespace ML;
using namespace MLCommon;


class TSNETest: public ::testing::Test {
protected:
    void basicTest() {

        cumlHandle handle;
        cudaStream_t stream; CUDA_CHECK(cudaStreamCreate(&stream));

        float *X_d, *Y_d;
        MLCommon::allocate(X_d, n*p);
        MLCommon::allocate(Y_d, n*2);
        MLCommon::updateDevice(X_d, digits.data(), n*p, stream);

        std::cout << "[>>>>]    Starting TSNE....\n";
        TSNE(handle, X_d, Y_d, n, p);
        std::cout << "[>>>>]    Got embeddings!....\n";

        std::cout << MLCommon::arr2Str(Y_d, n*2, "embeddings", stream) << std::endl;

        std::cout << "Updating host" << std::endl;
        float embeddings_h[n*2];
        MLCommon::updateHost(embeddings_h, Y_d, n*2, stream);

        std::cout << "DONE!" << std::endl;

        CUDA_CHECK(cudaFree(Y_d));
        CUDA_CHECK(cudaFree(X_d));

        std::cout << "End!" << std::endl;

        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    void SetUp() override {
        basicTest();
    }

    void TearDown() override {
    }

protected:
    int n = 1797;
    int p = 64;
};


typedef TSNETest TSNETestF;
TEST_F(TSNETestF, Result) {}

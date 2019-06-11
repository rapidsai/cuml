#include <cuda_utils.h>
#include <gtest/gtest.h>
#include <test_utils.h>
#include "ml_utils.h"
#include <random/rng.h>
#include <data_generators.h>

namespace ML {

using namespace MLCommon;

void test_function() {
  int random_state = 100;
  auto rng = Random::Rng(random_state);

  int n = 20;
  std::vector<float> x(n);
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
 
  rng.normal(x.data(), n, 0.0f, 1.0f, stream);
  
  CUDA_CHECK(cudaStreamDestroy(stream));
  for (int i = 0; i < n; i++) {
    std::cout << x[i] << ", ";
  }
    
  std::cout << "Demo" << std::endl;
}

void demo_class() {
  std::vector<float> data;
  std::vector<int> labels;
  
  makeClassificationDataHost(data, labels,
                             1000, 5, 3,
                             2, 2);
  
}

void demo_reg() {
  std::vector<float> X, y, coeff;
  
  makeRegressionDataHost(X, y, coeff,
                         100, 10, 5,
                         0.0f);
  myPrintHostVector("y", y.data(), y.size());
  myPrintHostMatrix("X", X.data(), 100, 10,
                    true, std::cout);
}


TEST(demo_rng, demo_reg) {
  demo_reg();
  ASSERT_EQ(1, 1);
}


// TEST(demo_rng, demo2) {
//   demo_class();
//   ASSERT_EQ(1, 1);
// }

}
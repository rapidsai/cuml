#include <gtest/gtest.h>
#include "cuda_utils.h"


namespace MLCommon {

TEST(Utils, Assert) {
    ASSERT_NO_THROW(ASSERT(1 == 1, "Should not assert!"));
    ASSERT_THROW(ASSERT(1 != 1, "Should assert!"), std::runtime_error);
}

TEST(Utils, CudaCheck) {
    ASSERT_NO_THROW(CUDA_CHECK(cudaFree(nullptr)));
}

// we want the functions like 'log2' to work both at compile and runtimes!
static const int log2Of1024 = log2(1024);
static const int log2Of1023 = log2(1023);
TEST(Utils, log2) {
    ASSERT_EQ(10, log2(1024));
    ASSERT_EQ(9, log2(1023));
    ASSERT_EQ(10, log2Of1024);
    ASSERT_EQ(9, log2Of1023);
}

} // end namespace MLCommon

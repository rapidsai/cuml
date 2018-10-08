#include <gtest/gtest.h>
#include "mnist.h"


namespace MLCommon {
namespace mnist {

TEST(Mnist, Parse) {
    ASSERT_EQ(0, system("../scripts/download_mnist.sh"));
    Dataset data("train-images-idx3-ubyte.gz",
                 "train-labels-idx1-ubyte.gz");
    ASSERT_EQ(60000, data.nImages);
    ASSERT_EQ(28, data.nRows);
    ASSERT_EQ(28, data.nCols);
}

} // end namespace mnist
} // end namespace MLCommon

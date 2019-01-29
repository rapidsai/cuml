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
#include "mnist.h"


namespace MLCommon {
namespace mnist {

TEST(Mnist, Parse) {
  ASSERT_EQ(0, system("../scripts/download_mnist.sh"));
  Dataset data("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz");
  ASSERT_EQ(60000, data.nImages);
  ASSERT_EQ(28, data.nRows);
  ASSERT_EQ(28, data.nCols);
}

} // end namespace mnist
} // end namespace MLCommon

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

#include <gtest/gtest.h>
#include <cuda_utils.h>
#include <test_utils.h>
#include "ml_utils.h"
#include "randomforest/randomforest.h"

namespace ML {

using namespace MLCommon;

template<typename T>
class RFTest: public ::testing::Test {
protected:
	void basicTest() {

		/* FIXME Placeholder
		   - Allocate input data set (rows, cols)
		   - Generate ref vals (somehow compare creted trees or predictions)?
		   - etc.
		*/
    }

 	void SetUp() override {
		basicTest();
	}

	void TearDown() override {
		//FIXME free any cuda allocs
		//CUDA_CHECK(cudaFree(d_ref_D));
	}

protected:

   	//placeholder for any params?
    int trees = 10;
	float * data;
    int * labels, * predictions, * ref_predictions;

    rfClassifier * rf_classifier = new rfClassifier(trees, 0, 0, RF_type::CLASSIFICATION);
};


//FIXME Add tests for fit and predict. Identify what would make a comparison match (similar predictions?)
typedef RFTest<float> RFTestF;
TEST_F(RFTestF, Fit) {
	//ASSERT_TRUE(
	//		devArrMatch(ref_predictions, predictions, /*some row cnt? */ , Compare<float>()));
}

} // end namespace ML

#include "knn/knn.cu"
#include <vector>
#include <gtest/gtest.h>
#include <cuda_utils.h>
#include <test_utils.h>
#include <iostream>

namespace ML {

using namespace MLCommon;


/**
 *
 * NOTE: Not exhaustively testing the kNN implementation since
 * we are using FAISS for this. Just testing API to verify the
 * knn.cu class is accepting inputs and providing outputs as
 * expected.
 */
template<typename T>
class KNN_MGTest: public ::testing::Test {
protected:
	void basicTest() {

		// Allocate input
        cudaSetDevice(0);
        allocate(d_train_inputs_dev1, n * d);
        cudaSetDevice(1);
        allocate(d_train_inputs_dev2, n * d);

        // Allocate reference arrays
        allocate<long>(d_ref_I, n*n);
        allocate(d_ref_D, n*n);

        // Allocate predicted arrays
        allocate<long>(d_pred_I, n*n);
        allocate(d_pred_D, n*n);

        // make test data on host
        std::vector<T> h_train_inputs = {1.0, 50.0, 51.0};
        h_train_inputs.resize(n);

        updateDevice(d_train_inputs_dev1, h_train_inputs.data(), n*d);
        updateDevice(d_train_inputs_dev2, h_train_inputs.data(), n*d);

        std::vector<T> h_res_D = { 0.0, 0.0, 2401.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0 };
        h_res_D.resize(n*n);
        updateDevice(d_ref_D, h_res_D.data(), n*n);

        std::vector<long> h_res_I = { 0, 3, 1, 1, 4, 2, 2, 5, 1 };
        h_res_I.resize(n*n);
        updateDevice<long>(d_ref_I, h_res_I.data(), n*n);

        params[0] = { d_train_inputs_dev1, n };
        params[1] = { d_train_inputs_dev2, n };

        knn->fit(params, 2);
        knn->search(d_train_inputs_dev1, n, d_pred_I, d_pred_D, n);
    }

 	void SetUp() override {
		basicTest();
	}

	void TearDown() override {
		CUDA_CHECK(cudaFree(d_train_inputs_dev1));
		CUDA_CHECK(cudaFree(d_train_inputs_dev2));
		CUDA_CHECK(cudaFree(d_pred_I));
		CUDA_CHECK(cudaFree(d_pred_D));
		CUDA_CHECK(cudaFree(d_ref_I));
		CUDA_CHECK(cudaFree(d_ref_D));
	}

protected:

	T* d_train_inputs_dev1;
	T* d_train_inputs_dev2;

    kNNParams *params = new kNNParams[2];

	int n = 3;
	int d = 1;

    long *d_pred_I;
    T* d_pred_D;

    long *d_ref_I;
    T* d_ref_D;

    kNN *knn = new kNN(d);
};


typedef KNN_MGTest<float> KNNTestF;
TEST_F(KNNTestF, Fit) {

	ASSERT_TRUE(
			devArrMatch(d_ref_D, d_pred_D, n*n, Compare<float>()));
	ASSERT_TRUE(
			devArrMatch(d_ref_I, d_pred_I, n*n, Compare<long>()));
}

} // end namespace ML
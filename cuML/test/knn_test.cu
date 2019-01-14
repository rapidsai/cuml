
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
class KNNTest: public ::testing::Test {
protected:
	void basicTest() {

		std::cout << "Allocating" << std::endl;

		// Allocate input
        allocate(d_train_inputs, n * d);

        // Allocate reference arrays
        allocate<long>(d_ref_I, n*n);
        allocate(d_ref_D, n*n);

        // Allocate predicted arrays
        allocate<long>(d_pred_I, n*n);
        allocate(d_pred_D, n*n);



        std::cout << "Building inputs" << std::endl;
        // make testdata on host
        std::vector<T> h_train_inputs = {1.0, 50.0, 51.0};
        h_train_inputs.resize(n);
        updateDevice(d_train_inputs, h_train_inputs.data(), n*d);

        std::vector<T> h_res_D = { 0.0, 2401.0, 2500.0, 0.0, 1.0, 2401.0, 0.0, 1.0, 2500.0 };
        h_res_D.resize(n*n);
        updateDevice(d_ref_D, h_res_D.data(), n*n);

        std::vector<long> h_res_I = { 0, 1, 2, 1, 2, 0, 2, 1, 0 };
        h_res_I.resize(n*n);
        updateDevice<long>(d_ref_I, h_res_I.data(), n*n);

        std::cout << "Fitting" << std::endl;
        knn->fit(d_train_inputs, n);
        std::cout << "Searching" << std::endl;

        knn->search(d_train_inputs, n, d_pred_I, d_pred_D, n);

        std::vector<float> h_output_D;
        h_output_D.resize(n*n);
        updateHost<float>(h_output_D.data(), d_pred_D, n*n);

        for(int i = 0; i < n*n; i++) {
        	std::cout << std::to_string(h_output_D[i]) << std::endl;
        }

        std::cout << "Done Searching" << std::endl;

    }

 	void SetUp() override {

 		std::cout << " Setting up!" << std::endl;

		basicTest();

		std::cout << "Done." << std::endl;
	}

	void TearDown() override {
		CUDA_CHECK(cudaFree(d_train_inputs));
		CUDA_CHECK(cudaFree(d_pred_I));
		CUDA_CHECK(cudaFree(d_pred_D));
		CUDA_CHECK(cudaFree(d_ref_I));
		CUDA_CHECK(cudaFree(d_ref_D));
	}

protected:

	T* d_train_inputs;

	int n = 3;
	int d = 1;

    long *d_pred_I;
    T* d_pred_D;

    long *d_ref_I;
    T* d_ref_D;

    kNN *knn = new kNN(d);
};


typedef KNNTest<float> KNNTestF;
TEST_F(KNNTestF, Fit) {
	ASSERT_TRUE(
			devArrMatch(d_ref_D, d_pred_D, n*n, Compare<float>()));
	ASSERT_TRUE(
			devArrMatch(d_ref_I, d_pred_I, n*n, Compare<long>()));
}

} // end namespace ML

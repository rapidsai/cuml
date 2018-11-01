
#include "kmeans/kmeans.cu"
#include <gtest/gtest.h>
#include <cuda_utils.h>
#include <test_utils.h>

namespace ML {

using namespace MLCommon;

template<typename T>
struct KmeansInputs {
    int n_clusters;
	T tol;
	int n_row;
	int n_col;
};

template<typename T>
class KmeansTest: public ::testing::TestWithParam<KmeansInputs<T> > {
protected:
	void basicTest() {
		params = ::testing::TestWithParam<KmeansInputs<T>>::GetParam();
        int m = params.n_row;
        int n = params.n_col;
        int k = params.n_clusters;

        // make testdata on host
		T h_srcdata[n * m] = 
            {1.0,1.0,3.0,4.0, 1.0,2.0,2.0,3.0};

        // make space for outputs : pred_centroids, pred_labels
        // and reference output : labels_ref
   		allocate(labels_fit, m);
   		allocate(labels_ref_fit, m);
        allocate(pred_centroids, k * n);
        allocate(centroids_ref, k * n);

        // make and assign reference output
        int h_labels_ref_fit[m] = {1, 1, 0, 0};
        updateDevice(labels_ref_fit, h_labels_ref_fit, m);

        T h_centroids_ref[k * n] = {3.5,2.5, 1.0,1.5};
        updateDevice(centroids_ref, h_centroids_ref, k * n);

        // The actual kmeans api calls
        // fit
        make_ptr_kmeans(0, verbose, seed, gpu_id, n_gpu, m, n,
            ord, k, k, max_iterations,
            init_from_data, params.tol, h_srcdata, nullptr, pred_centroids, labels_fit);
    }

 	void SetUp() override {
		basicTest();
	}

	void TearDown() override {
		CUDA_CHECK(cudaFree(labels_fit));
		CUDA_CHECK(cudaFree(pred_centroids));
		CUDA_CHECK(cudaFree(labels_ref_fit));
		CUDA_CHECK(cudaFree(centroids_ref));

	}

protected:
	KmeansInputs<T> params;
	T *d_srcdata;
	int *labels_fit, *labels_ref_fit;
    T *pred_centroids, *centroids_ref;
    int verbose = 0;
    int seed = 1;
    int gpu_id = 0;
    int n_gpu = -1;
    char ord = 'c'; // here c means col order, NOT C (vs F) order
    int max_iterations = 300;
    int init_from_data = 0;
};

const std::vector<KmeansInputs<float> > inputsf2 = {
		{ 2, 0.05f, 4, 2 }};

const std::vector<KmeansInputs<double> > inputsd2 = {
		{ 2, 0.05, 4, 2 }};

typedef KmeansTest<float> KmeansTestF;
TEST_P(KmeansTestF, Fit) {
	ASSERT_TRUE(
			devArrMatch(labels_ref_fit, labels_fit, params.n_row,
					CompareApproxAbs<float>(params.tol)));
	ASSERT_TRUE(
			devArrMatch(centroids_ref, pred_centroids, params.n_clusters * params.n_col,
					CompareApproxAbs<float>(params.tol)));
}

typedef KmeansTest<double> KmeansTestD;
TEST_P(KmeansTestD, Fit) {
	ASSERT_TRUE(
			devArrMatch(labels_ref_fit, labels_fit, params.n_row,
					CompareApproxAbs<double>(params.tol)));
	ASSERT_TRUE(
			devArrMatch(centroids_ref, pred_centroids, params.n_clusters * params.n_col,
					CompareApproxAbs<double>(params.tol)));
}

INSTANTIATE_TEST_CASE_P(KmeansTests, KmeansTestF, ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(KmeansTests, KmeansTestD, ::testing::ValuesIn(inputsd2));

} // end namespace ML

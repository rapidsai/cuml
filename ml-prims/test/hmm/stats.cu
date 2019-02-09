#include <gtest/gtest.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "hmm/stats.h"
#include "hmm/utils.h"
#include "cuda_utils.h"

#define IDX(i,j,lda) ((i)+(j)*(lda))
#define IDX2(i,j,k,lda,ldb) (IDX(i,j,lda) + k * ldb)

using namespace MLCommon::LinAlg;
using namespace MLCommon;

namespace MLCommon {
namespace HMM {


template <typename T>
T _naive_mean(T* weights, T* data, int dim, int clId, int dims, int nPts, int nCl){
        T mean = 0;
        for (int i = 0; i < nPts; ++i)
                mean += weights[IDX(clId, i, nCl)] * data[IDX(dim, i, dims)];
        return mean;
}

template <typename T>
void naive_means(T* weights_h, T* data_h, T* means_h, int dim, int nPts, int nCl) {
        for (int i = 0; i < dim; ++i)
                for (int clId = 0; clId < nCl; ++clId)
                        means_h[IDX(i, clId, dim)] = _naive_mean(weights_h, data_h, i, clId, dim, nPts, nCl);
}



// template <typename T>
// T _dot_diff(T* x, T* mu_a, T* mu_b, int n_dim){
//         T dot = 0;
//         for (int dimId = 0; dimId < n_dim; ++dimId)
//                 dot += (x[dimId] - mu_a[dimId]) * (x[dimId] - mu_b[dimId]);
//         return dot;
// }

template <typename T>
T _naive_cov(int row, int col, int cl, T* means, T* weights, T* data, int nDims, int nPts, int nCl){
        T cov = 0;
        for (int ptId = 0; ptId < nPts; ++ptId)
                cov += weights[IDX(cl, ptId, nCl)] *
                       (data[IDX(row, ptId, nDims)] - means[IDX(row, cl, nDims)]) * (data[IDX(col, ptId, nDims)] - means[IDX(col, cl, nDims)]);
        // cov += weights[IDX(cl, ptId, nCl)] *
        //        _dot_diff(data + IDX(0, ptId, nDims),
        //                  means + IDX(0, row, nDims),
        //                  means + IDX(0, col, nDims),
        //                  nDims
        //                  );
        return cov;
}

template <typename T>
void naive_covs(T* sigmas, T* weights, T* data, T* means, int nDims, int nPts, int nCl) {
        for (int col = 0; col < nDims; ++col)
                for (int row = 0; row < nDims; ++row)
                        for (int clId = 0; clId < nCl; ++clId)
                                sigmas[IDX2(row, col, clId, nDims, nDims*nDims)] = _naive_cov(row, col, clId, means, weights, data, nDims, nPts, nCl);
}

template <typename T>
T _compute_error(T* true_arr, T* est_arr, int arr_size){
        T diff = 0;
        for (int i = 0; i < arr_size; i++)
                diff += std::abs(true_arr[i] - est_arr[i]);
        diff /= arr_size;
        return diff;
}

template <typename T>
T compute_error_means(T *data_d, T *weights_d, T *data_h, T *weights_h, int dim,
                      int nPts, int nCl, cublasHandle_t handle){
        T error;
        T* true_means;
        T *comp_means_d, *comp_means_h;

        true_means = (T *)malloc(dim * nCl * sizeof(T));
        comp_means_h = (T *)malloc(dim * nCl * sizeof(T));
        allocate(comp_means_d, dim * nCl);

        naive_means(weights_h, data_h, true_means, dim, nPts, nCl);
        weighted_means(weights_d, data_d, comp_means_d, dim, nPts, nCl, handle);

        updateHost(comp_means_h, comp_means_d, dim * nCl);

        int array_size = dim * nCl;
        error = _compute_error(true_means, comp_means_h, array_size);
        return error;
}

template <typename T>
T compute_error_covs(T *data_d, T *weights_d, T *means_d,
                     T *data_h, T *weights_h, T *means_h,
                     int nDims, int nPts, int nCl,
                     cublasHandle_t handle){
        T error;
        T* true_covs;
        T *comp_covs_d, *comp_covs_h;

        true_covs = (T *)malloc(nDims * nDims * nCl * sizeof(T));
        comp_covs_h = (T *)malloc(nDims * nDims * nCl * sizeof(T));
        allocate(comp_covs_d, nDims * nDims * nCl);

        naive_covs(true_covs, weights_h, data_h, means_h, nDims, nPts, nCl);

        weighted_covs(data_d, weights_d, means_d, comp_covs_d,
                      nDims, nPts, nCl, &handle);

        updateHost(comp_covs_h, comp_covs_d, nDims * nDims * nCl);

        print_matrix_host(weights_h, nCl, nPts, "weights");
        print_matrix_host(data_h, nDims, nPts, "data");
        print_matrix_host(true_covs, nDims * nDims, nCl, "true covs");
        print_matrix_host(comp_covs_h, nDims * nDims, nCl, "comp covs");

        int array_size = nDims * nDims * nCl;
        error = _compute_error(true_covs, comp_covs_h, array_size);
        return error;
}

template <typename T>
struct StatsInputs {
        T tolerance;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const StatsInputs<T>& dims) {
        return os;
}

template <typename T>
class StatsTest : public ::testing::TestWithParam<StatsInputs<T> > {
protected:
void SetUp() override {
        params = ::testing::TestWithParam<StatsInputs<T> >::GetParam();
        tolerance = params.tolerance;

        cublasHandle_t handle;
        CUBLAS_CHECK(cublasCreate(&handle));

        data_h = (T *)malloc(dim * nPts * sizeof(T));
        weights_h = (T *)malloc(nPts * nCl * sizeof(T));
        means_h = (T *)malloc(dim * nCl * sizeof(T));
        allocate(data_d, dim * nPts);
        allocate(weights_d, nCl * nPts);
        allocate(means_d, nCl * dim);

        CUDA_CHECK(cudaMemset(weights_d, 0, nCl * nPts));
        CUDA_CHECK(cudaMemset(data_d, 0, nCl * nPts));
        CUDA_CHECK(cudaMemset(means_d, 0, nCl * dim));

        data_h[0] = 5.0;
        data_h[1] = 0.0;
        data_h[2] = 6.0;
        data_h[3] = 10.0;
        data_h[4] = -1.0;
        data_h[5] = -7.0;
        data_h[6] = 0.0;
        data_h[7] = -3.0;
        data_h[8] = -15.0;
        data_h[9] = 20.0;

        weights_h[0] = 0.20;
        weights_h[1] = 0.40;
        weights_h[2] = 0.0;

        weights_h[3] = 0.20;
        weights_h[4] = 0.15;
        weights_h[5] = 0.0;

        weights_h[6] = 0.2;
        weights_h[7] = 0.15;
        weights_h[8] = 0.0;

        weights_h[9] = 0.2;
        weights_h[10] = 0.15;
        weights_h[11] = 0.0;

        weights_h[12] = 0.2;
        weights_h[13] = 0.15;
        weights_h[14] = 1.0;

        // copy data to gpu (available in ml-common/cuda_utils.h)
        updateDevice(data_d, data_h, dim * nPts);
        updateDevice(weights_d, weights_h, nCl * nPts);
        T error_means = compute_error_means(data_d, weights_d, data_h, weights_h, dim, nPts, nCl, handle);

        weighted_means(weights_d, data_d, means_d, dim, nPts, nCl, handle);
        updateHost(means_h, means_d, dim * nCl);
        T error_covs = compute_error_covs(data_d, weights_d, means_d, data_h,
                                          weights_h, means_h, dim, nPts, nCl, handle);
        error = error_means + error_covs;
}

void TearDown() override {
        free(data_h);
        free(weights_h);
        free(means_h);

        CUDA_CHECK(cudaFree(data_d));
        CUDA_CHECK(cudaFree(weights_d));
        CUDA_CHECK(cudaFree(means_d));
}

protected:
StatsInputs<T> params;
T *data_h, *weights_h, *data_d, *weights_d;
T *means_d, *means_h;
int dim = 2;
int nCl = 3;
int nPts = 5;
T error, tolerance;
};

const std::vector<StatsInputs<float> > inputsf2 = {
        {0.00001f}
};

const std::vector<StatsInputs<double> > inputsd2 = {
        {0.00001}
};


typedef StatsTest<float> StatsTestF;
TEST_P(StatsTestF, Result){
        EXPECT_LT(error, tolerance) << " error out of tol.";
}

typedef StatsTest<double> StatsTestD;
TEST_P(StatsTestD, Result){
        EXPECT_LT(error, tolerance) << " error out of tol.";
}

INSTANTIATE_TEST_CASE_P(StatsTests, StatsTestF,
                        ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(StatsTests, StatsTestD,
                        ::testing::ValuesIn(inputsd2));

} // end namespace HMM
} // end namespace MLCommon

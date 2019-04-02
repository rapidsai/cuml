#include <gtest/gtest.h>

// #include <magma_v2.h>
#include "gmm/gmm.h"
#include <magma/magma_test_utils.h>

using namespace MLCommon;
using namespace gmm;

template <typename T>
struct GMMInputs {
        T tolerance;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const GMMInputs<T>& dims) {
        return os;
}

template <typename T>
void run(magma_int_t nCl, magma_int_t nDim, magma_int_t nObs, int n_iter)
{
        T *dX;
        // declaration:
        T *dmu, *dsigma, *dPis, *dPis_inv, *dLlhd;
        magma_int_t lddx, lddmu, lddsigma, lddsigma_full, lddPis, lddLlhd;
        lddx = magma_roundup(nDim, RUP_SIZE);
        lddmu = magma_roundup(nDim, RUP_SIZE);
        lddsigma = magma_roundup(nDim, RUP_SIZE);
        lddsigma_full = nDim * lddsigma;
        lddLlhd = magma_roundup(nCl, RUP_SIZE);
        lddPis = lddLlhd;

        // Random parameters
        T start=0;
        T end = 1;
        unsigned long long seed = 1234ULL;

        cublasHandle_t cublasHandle;
        CUBLAS_CHECK(cublasCreate(&cublasHandle));

        int device = 0;
        magma_queue_t queue;
        magma_queue_create(device, &queue);

        allocate(dX, lddx * nObs);
        allocate(dmu, lddmu * nCl);
        allocate(dsigma, lddsigma_full * nCl);
        allocate(dLlhd, lddLlhd * nObs);
        allocate(dPis, lddPis);
        allocate(dPis_inv, lddPis);

//         load_csv(dX, );
//         load_csv(dmu, );
//         load_csv(dsigma,);
//         load_csv(dLlhd, );
//         load_csv(dPis, );
//         load_csv(dPis_inv, );
//
// // computation:
//         GMM<T> gmm;
//         init(gmm,
//              dmu, dsigma, dPis, dPis_inv, dLlhd,
//              lddx, lddmu, lddsigma, lddsigma_full, lddPis, lddLlhd,
//              cur_llhd, reg_covar,
//              nCl, nDim, nObs);
//         setup(gmm);
//         fit(dX, n_iter, gmm, cublasHandle, queue);

// cleanup:
        // CUDA_CHECK(cudaFree(dX));
}


template <typename T>
class GMMTest : public ::testing::TestWithParam<GMMInputs<T> > {
protected:
void SetUp() override {
        params = ::testing::TestWithParam<GMMInputs<T> >::GetParam();
        tolerance = params.tolerance;

        magma_int_t nCl = 2;
        magma_int_t nDim = 3;
        magma_int_t nObs = 4;
        int n_iter = 5;

        run<T>(nCl, nDim, nObs, n_iter);

        error = 1;

        magma_finalize();
}

void TearDown() override {
}

protected:
GMMInputs<T> params;
T error, tolerance;
};

const std::vector<GMMInputs<float> > inputsf2 = {
        {0.00001f}
};

const std::vector<GMMInputs<double> > inputsd2 = {
        {0.00001}
};


typedef GMMTest<float> GMMTestF;
TEST_P(GMMTestF, Result){
        EXPECT_LT(error, tolerance) << " error out of tol.";
}

typedef GMMTest<double> GMMTestD;
TEST_P(GMMTestD, Result){
        EXPECT_LT(error, tolerance) << " error out of tol.";
}

INSTANTIATE_TEST_CASE_P(GMMTests, GMMTestF,
                        ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(GMMTests, GMMTestD,
                        ::testing::ValuesIn(inputsd2));

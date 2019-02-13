#include <gtest/gtest.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "hmm/likelihood.h"
#include "hmm/utils.h"

#include "cuda_utils.h"


using namespace MLCommon::LinAlg;
using namespace MLCommon;

namespace MLCommon {
namespace HMM {


template <typename T>
struct LikelihoodInputs {
        T tolerance;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const LikelihoodInputs<T>& dims) {
        return os;
}

template <typename T>
class LikelihoodTest : public ::testing::TestWithParam<LikelihoodInputs<T> > {
protected:
void SetUp() override {
        params = ::testing::TestWithParam<LikelihoodInputs<T> >::GetParam();
        tolerance = params.tolerance;

        initialize_parameters();
        allocate_memory();
        initialize();
        copy_to_device();
        error = compute_error_lhd();
}

T compute_error_lhd(){
        T diff=0;
        T est_val, true_val;
        for (int testId = 0; testId < nTests; testId++) {
                est_val = set_gmm_lhd(data_d, mus_d, sigmas_d,
                                      rhos_d + nDim * testId,
                                      isLog, nCl, nDim, nObs, &handle);
                true_val = *(llhds_h + testId);
                diff += std::abs(true_val - est_val);
        }
        diff /= nTests;
        return diff;
}

void allocate_memory(){
        allocate(data_d, nDim * nObs);
        CUDA_CHECK(cudaMemset(data_d, (T)0, nDim * nObs ));
        data_h = (T *)malloc(nDim * nObs * sizeof(T));

        allocate(mus_d, nDim * nCl);
        CUDA_CHECK(cudaMemset(mus_d, (T)0,nDim * nCl ));
        mus_h = (T *)malloc(nDim * nCl * sizeof(T));

        allocate(sigmas_d, nDim *nDim * nCl);
        CUDA_CHECK(cudaMemset(sigmas_d, (T) 0, nDim *nDim * nCl));
        sigmas_h = (T *)malloc(nDim * nDim * nCl * sizeof(T));

        allocate(rhos_d, nTests * nCl);
        CUDA_CHECK(cudaMemset(rhos_d, (T) 0, nTests * nCl));
        rhos_h = (T *)malloc(nTests * nCl * sizeof(T));

        allocate(llhds_d, nTests);
        CUDA_CHECK(cudaMemset(llhds_d, (T) 0, nTests));
        llhds_h = (T *)malloc(nTests * sizeof(T));

        CUBLAS_CHECK(cublasCreate(&handle));
}

void initialize_parameters(){
        nDim = 2;
        nCl = 2;
        nObs = 1;
        nDists = 1;
        isLog = true;
        nTests = 3;
}

void initialize(){

        data_h[0] = 1.0;
        data_h[1] = -0.75;
        // data_h[2] = 1.0;
        // data_h[3] = 5.0;

        mus_h[0] = 0.0;
        mus_h[1] = 0.0;
        mus_h[2] = 1.0;
        mus_h[3] = 0.65;

        rhos_h[0] = 1.0;
        rhos_h[1] = 0.0;
        rhos_h[2] = 0.0;
        rhos_h[3] = 1.0;
        rhos_h[4] = 0.3;
        rhos_h[5] = 0.7;

        sigmas_h[0] = 1;
        sigmas_h[1] = 0.0;
        sigmas_h[2] = 0.0;
        sigmas_h[3] = 1.0;
        sigmas_h[4] = 1.0;
        sigmas_h[5] = 0.5;
        sigmas_h[6] = 0.5;
        sigmas_h[7] = 2.0;

        llhds_h[0] = -2.6191270664093453;
        llhds_h[1] = -2.677684960377056;
        llhds_h[2] = -2.6601175921867424;
}

void copy_to_device(){
        updateDevice(data_d, data_h, nDim * nObs);
        updateDevice(mus_d, mus_h, nDim * nCl);
        updateDevice(sigmas_d, sigmas_h, nDim *nDim * nCl);
        updateDevice(rhos_d, rhos_h, nCl * nTests);
        updateDevice(llhds_d, llhds_h, nTests);
}



void TearDown() override {
        free(data_h);
        free(mus_h);
        free(sigmas_h);
        free(rhos_h);
        free(llhds_h);

        CUDA_CHECK(cudaFree(data_d));
        CUDA_CHECK(cudaFree(mus_d));
        CUDA_CHECK(cudaFree(sigmas_d));
        CUDA_CHECK(cudaFree(rhos_d));
        CUDA_CHECK(cudaFree(llhds_d));
}

protected:
LikelihoodInputs<T> params;
T error, tolerance;
int nDim, nCl, nObs, nDists;
bool isLog;
int nTests;

T *llhds_h, *llhds_d;
T *data_h, *mus_h, *sigmas_h, *rhos_h;
T *data_d, *mus_d, *sigmas_d, *rhos_d;
cublasHandle_t handle;
};

const std::vector<LikelihoodInputs<float> > inputsf2 = {
        {0.00001f}
};

const std::vector<LikelihoodInputs<double> > inputsd2 = {
        {0.00001}
};


typedef LikelihoodTest<float> LikelihoodTestF;
TEST_P(LikelihoodTestF, Result){
        EXPECT_LT(error, tolerance) << " error out of tol.";
}

typedef LikelihoodTest<double> LikelihoodTestD;
TEST_P(LikelihoodTestD, Result){
        EXPECT_LT(error, tolerance) << " error out of tol.";
}

INSTANTIATE_TEST_CASE_P(LikelihoodTests, LikelihoodTestF,
                        ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(LikelihoodTests, LikelihoodTestD,
                        ::testing::ValuesIn(inputsd2));

} // end namespace HMM
} // end namespace MLCommon

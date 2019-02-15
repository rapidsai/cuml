#include <gtest/gtest.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "hmm/utils.h"
#include "hmm/gmm.cu"

#include "linalg/cusolver_wrappers.h"
#include "linalg/mean_squared_error.h"
#include "cuda_utils.h"


using namespace MLCommon::LinAlg;
using namespace MLCommon;
using namespace ML::HMM;

namespace MLCommon {
namespace HMM {


template <typename T>
struct GMMInputs {
        T tolerance;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const GMMInputs<T>& dims) {
        return os;
}

template <typename T>
class GMMTest : public ::testing::TestWithParam<GMMInputs<T> > {
protected:
void SetUp() override {
        params = ::testing::TestWithParam<GMMInputs<T> >::GetParam();
        tolerance = params.tolerance;

        setup_gmm();
        printf("after setup iterations %d\n", gmm.paramsEm->n_iter);

        allocate_memory();
        printf("alloc iterations %d\n", gmm.paramsEm->n_iter);

        // initialize_data();
        printf("iniit iterations %d\n", gmm.paramsEm->n_iter);
        // copy_to_device();
        // printf(" befroe fit iterations %d\n", gmm.paramsEm->n_iter);
        // fit(gmm, data_d);
        // compute_error(gmm);
}

void compute_error(GMM<T>& gmm){
        error = 0.0;

        print_matrix( gmm.mus, nDim, 1, "gmm mus");
        print_matrix( mus_d, nDim, 1, "mus");

        meanSquaredError(error_d, sigmas_d, gmm.sigmas, nDim * nDim * nCl);
        updateHost(&temp_err, error_d, 1);
        error += temp_err;

        meanSquaredError(error_d, rhos_d, gmm.rhos, nCl);
        updateHost(&temp_err, error_d, 1);
        error += temp_err;
}

void setup_gmm(){
        nDim = 1;
        nCl = 2;
        nObs = 2;

        n_iter = 10;

        paramsEM paramsEm(n_iter);
        paramsRandom<T> paramsRd((int) 0, (int) 1, (unsigned long long) 1234ULL);

        set_gmm(gmm, nCl, nDim, nObs, &paramsRd, &paramsEm);

        printf("iterations %d\n", gmm.paramsEm->n_iter);

        initialize(gmm);
}


void allocate_memory(){
        allocate(data_d, nDim * nObs);
        CUDA_CHECK(cudaMemset(data_d, (T)0, nDim * nObs ));

        allocate(mus_d, nDim * nCl);
        CUDA_CHECK(cudaMemset(mus_d, (T)0, nDim * nCl));

        allocate(sigmas_d, nDim *nDim * nObs);
        CUDA_CHECK(cudaMemset(sigmas_d, (T)0, nDim * nDim * nCl ));

        allocate(rhos_d, nCl);
        CUDA_CHECK(cudaMemset(rhos_d, (T)0, nCl ));

        data_h = (T *)malloc(nDim * nObs * sizeof(T));
        mus_h = (T *)malloc(nDim * nCl * sizeof(T));
        sigmas_h = (T *)malloc(nDim * nDim * nCl * sizeof(T));
        rhos_h = (T *)malloc(nCl * sizeof(T));

        allocate(error_d, 1);
}


void initialize_data(){


        // T data[] = {(T) 1.00958873,(T) -1.16595535,(T) -0.98994148,(T) -0.84248056,(T) -0.97283954,
        // (T)-1.1479895, (T)-1.00086317,(T) -1.03972452, (T)-0.93357705,(T) -1.0343065};

        // printf("t data iterations %d\n", );

        // for (size_t i = 0; i < 10; i++) {
        // data_h[i] = data[i];
        // }
        data_h[0] = (T) 0.0;
        data_h[1] = (T) -1.0;
        // data_h[2] = 1.0;
        // data_h[3] = 5.0;
        // data_h[4] = 0.0;
        // data_h[5] = -1.0;
        // data_h[6] = 1.0;
        // data_h[7] = 5.0;
        // data_h[8] = 0.0;
        // data_h[9] = -1.0;

        mus_h[0] = -1.0;
        mus_h[1] = 2.0;

        rhos_h[0] = 1;
        rhos_h[1] = 0;

        sigmas_h[0] = 0.1;
        sigmas_h[1] = 0.1;

}

void copy_to_device(){
        printf(" value %d line number %d in file %s\n", (int) gmm.paramsEm->n_iter, __LINE__, __FILE__);
        updateDevice(data_d, data_h, nDim * nObs);
        printf(" value %d line number %d in file %s\n", (int) gmm.paramsEm->n_iter, __LINE__, __FILE__);
        updateDevice(mus_d, mus_h, nDim * nCl);
        printf(" value %d line number %d in file %s\n", (int) gmm.paramsEm->n_iter, __LINE__, __FILE__);
        updateDevice(sigmas_d, sigmas_h, nDim *nDim * nCl);
        printf(" value %d line number %d in file %s\n", (int) gmm.paramsEm->n_iter, __LINE__, __FILE__);
        updateDevice(rhos_d, rhos_h, nCl);
}

void TearDown() override {
        free(data_h);
        free(mus_h);
        free(sigmas_h);
        free(rhos_h);

        CUDA_CHECK(cudaFree(data_d));
        CUDA_CHECK(cudaFree(mus_d));
        CUDA_CHECK(cudaFree(sigmas_d));
        CUDA_CHECK(cudaFree(rhos_d));

        CUDA_CHECK(cudaFree(error_d));
}

protected:
GMM<T> gmm;

GMMInputs<T> params;
T error, tolerance;
int nDim, nCl, nObs;

int n_iter;

T *data_h, *mus_h, *sigmas_h, *rhos_h;
T *data_d, *mus_d, *sigmas_d, *rhos_d;

cusolverDnHandle_t cusolverHandle;

T *error_d, temp_err;
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

} // end namespace HMM
} // end namespace MLCommon

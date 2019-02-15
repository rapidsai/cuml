#include <gtest/gtest.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "hmm/determinant.h"
#include "hmm/likelihood.h"
// #include "hmm/utils.h"

#include "cuda_utils.h"


using namespace MLCommon::LinAlg;
using namespace MLCommon;

namespace MLCommon {
namespace HMM {


template <typename T>
struct DeterminantInputs {
        T tolerance;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const DeterminantInputs<T>& dims) {
        return os;
}

template <typename T>
class DeterminantTest : public ::testing::TestWithParam<DeterminantInputs<T> > {
protected:
void SetUp() override {
        params = ::testing::TestWithParam<DeterminantInputs<T> >::GetParam();
        initialize(params);
        copy_to_device();
        compute_error_det(true_det);
}

void initialize(DeterminantInputs<T> params){
        tolerance = params.tolerance;
        nDim = 2;

        M_h = (T *)malloc(nDim * nDim * sizeof(T));
        allocate(M_d, nDim * nDim);
        CUDA_CHECK(cudaMemset(M_d, (T)0, nDim * nDim));

        M_h[0] = (T) 9.5;
        M_h[1] = (T) 5;
        M_h[2] = (T) 5.;
        M_h[3] = (T) 4.;
        true_det = (T) 13;
}


void copy_to_device(){
        updateDevice(M_d, M_h, nDim * nDim);
}


void compute_error_det(T true_det){
        Determinant<T> determinant(nDim);
        T est_det = determinant.compute(M_d);
        // printf("line number %d in file %s\n", __LINE__, __FILE__);
        error = std::abs(est_det - true_det);
}


void TearDown() override {
        free(M_h);
        CUDA_CHECK(cudaFree(M_d));
}

protected:
DeterminantInputs<T> params;
T error, tolerance;
T true_det;
int nDim;

T *M_h, *M_d;
};


const std::vector<DeterminantInputs<float> > inputsf2 = {
        {0.00001f}
};

const std::vector<DeterminantInputs<double> > inputsd2 = {
        {0.00001}
};


typedef DeterminantTest<float> DeterminantTestF;
TEST_P(DeterminantTestF, Result){
        EXPECT_LT(error, tolerance) << " error out of tol.";
}

typedef DeterminantTest<double> DeterminantTestD;
TEST_P(DeterminantTestD, Result){
        EXPECT_LT(error, tolerance) << " error out of tol.";
}

INSTANTIATE_TEST_CASE_P(DeterminantTests, DeterminantTestF,
                        ::testing::ValuesIn(inputsf2));

INSTANTIATE_TEST_CASE_P(DeterminantTests, DeterminantTestD,
                        ::testing::ValuesIn(inputsd2));

} // end namespace HMM
} // end namespace MLCommon

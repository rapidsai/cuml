#include "random/rng.h"
#include "cuda_utils.h"
#include "test_utils.h"
#include <cub/cub.cuh>
#include <gtest/gtest.h>


namespace MLCommon {
namespace Random {

enum RandomType {
    RNG_Normal,
    RNG_LogNormal,
    RNG_Uniform,
    RNG_Gumbel,
    RNG_Logistic,
    RNG_Exp,
    RNG_Rayleigh,
    RNG_Laplace
};

template <typename T, int TPB>
__global__ void meanKernel(T* out, const T* data, int len) {
    typedef cub::BlockReduce<T, TPB> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    T val = tid < len? data[tid] : T(0);
    T x = BlockReduce(temp_storage).Sum(val);
    __syncthreads();
    T xx = BlockReduce(temp_storage).Sum(val*val);
    __syncthreads();
    if(threadIdx.x == 0) {
        myAtomicAdd(out, x);
        myAtomicAdd(out+1, xx);
    }
}

template <typename T>
struct RngInputs {
    T tolerance;
    int len;
    // start, end: for uniform
    // mean, sigma: for normal/lognormal
    // mean, beta: for gumbel
    // mean, scale: for logistic and laplace
    // lambda: for exponential
    // sigma: for rayleigh
    T start, end;
    RandomType type;
    unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const RngInputs<T>& dims) {
    return os;
}

template <typename T>
class RngTest: public ::testing::TestWithParam<RngInputs<T> > {
protected:
    void SetUp() override {
        params = ::testing::TestWithParam<RngInputs<T>>::GetParam();
        Rng<T> r(params.seed);
        allocate(data, params.len);
        allocate(stats, 2, true);
        switch(params.type) {
        case RNG_Normal:
            r.normal(data, params.len, params.start, params.end);
            break;
        case RNG_LogNormal:
            r.lognormal(data, params.len, params.start, params.end);
            break;
        case RNG_Uniform:
            r.uniform(data, params.len, params.start, params.end);
            break;
        case RNG_Gumbel:
            r.gumbel(data, params.len, params.start, params.end);
            break;
        case RNG_Logistic:
            r.logistic(data, params.len, params.start, params.end);
            break;
        case RNG_Exp:
            r.exponential(data, params.len, params.start);
            break;
        case RNG_Rayleigh:
            r.rayleigh(data, params.len, params.start);
            break;
        case RNG_Laplace:
            r.laplace(data, params.len, params.start, params.end);
            break;
        };
        static const int threads = 128;
        meanKernel<T,threads><<<ceildiv(params.len,threads),threads>>>
            (stats, data, params.len);
        updateHost<T>(h_stats, stats, 2);
        h_stats[0] /= params.len;
        h_stats[1] = (h_stats[1] / params.len) - (h_stats[0] * h_stats[0]);
    }

    void TearDown() override {
        CUDA_CHECK(cudaFree(data));
        CUDA_CHECK(cudaFree(stats));
    }

    void getExpectedMeanVar(T meanvar[2]) {
        switch(params.type) {
        case RNG_Normal:
            meanvar[0] = params.start;
            meanvar[1] = params.end * params.end;
            break;
        case RNG_LogNormal: {
            auto var = params.end * params.end;
            auto mu = params.start;
            meanvar[0] = myExp(mu + var * T(0.5));
            meanvar[1] = (myExp(var) - T(1.0)) * myExp(T(2.0)*mu+var);
            break;
        }
        case RNG_Uniform:
            meanvar[0] = (params.start + params.end) * T(0.5);
            meanvar[1] = params.end - params.start;
            meanvar[1] = meanvar[1] * meanvar[1] / T(12.0);
            break;
        case RNG_Gumbel: {
            auto gamma = T(0.577215664901532);
            meanvar[0] = params.start + params.end * gamma;
            meanvar[1] = T(3.1415) * T(3.1415) * params.end * params.end / T(6.0);
            break;
        }
        case RNG_Logistic:
            meanvar[0] = params.start;
            meanvar[1] = T(3.1415) * T(3.1415) * params.end * params.end / T(3.0);
            break;
        case RNG_Exp:
            meanvar[0] = T(1.0) / params.start;
            meanvar[1] = meanvar[0] * meanvar[0];
            break;
        case RNG_Rayleigh:
            meanvar[0] = params.start * mySqrt(T(3.1415 / 2.0));
            meanvar[1] = ((T(4.0) - T(3.1415)) / T(2.0)) * params.start * params.start;
            break;
        case RNG_Laplace:
            meanvar[0] = params.start;
            meanvar[1] = T(2.0) * params.end * params.end;
            break;
        };
    }

protected:
    RngInputs<T> params;
    T *data, *stats;
    T h_stats[2]; // mean, var
};

typedef RngTest<float> RngTestF;
const std::vector<RngInputs<float> > inputsf = {
    {0.001f, 32*1024,  1.f, 1.f,    RNG_Normal, 1234ULL},
    {0.005f,  8*1024,  1.f, 1.f,    RNG_Normal, 1234ULL},
    {0.05f,  32*1024,  1.f, 1.f, RNG_LogNormal, 1234ULL},
    {0.05f,   8*1024,  1.f, 1.f, RNG_LogNormal, 1234ULL},
    {0.001f, 32*1024, -1.f, 1.f,   RNG_Uniform, 1234ULL},
    {0.001f,  8*1024, -1.f, 1.f,   RNG_Uniform, 1234ULL},
    {0.01f,  32*1024,  1.f, 1.f,    RNG_Gumbel, 1234ULL},
    {0.05f,   8*1024,  1.f, 1.f,    RNG_Gumbel, 1234ULL},
    {0.01f,  32*1024,  1.f, 1.f,  RNG_Logistic, 1234ULL},
    {0.05f,   8*1024,  1.f, 1.f,  RNG_Logistic, 1234ULL},
    {0.01f,  32*1024,  1.f, 1.f,       RNG_Exp, 1234ULL},
    {0.05f,   8*1024,  1.f, 1.f,       RNG_Exp, 1234ULL},
    {0.01f,  32*1024,  1.f, 1.f,  RNG_Rayleigh, 1234ULL},
    {0.05f,   8*1024,  1.f, 1.f,  RNG_Rayleigh, 1234ULL},
    {0.01f,  32*1024,  1.f, 1.f,   RNG_Laplace, 1234ULL},
    {0.05f,   8*1024,  1.f, 1.f,   RNG_Laplace, 1234ULL}
};
TEST_P(RngTestF, Result) {
    float meanvar[2];
    getExpectedMeanVar(meanvar);
    ASSERT_TRUE(match(meanvar[0], h_stats[0], CompareApprox<float>(params.tolerance)));
    ASSERT_TRUE(match(meanvar[1], h_stats[1], CompareApprox<float>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(RngTests, RngTestF, ::testing::ValuesIn(inputsf));

typedef RngTest<double> RngTestD;
const std::vector<RngInputs<double> > inputsd = {
    {0.001,  32*1024,  1.0, 1.0,    RNG_Normal, 1234ULL},
    {0.005,   8*1024,  1.0, 1.0,    RNG_Normal, 1234ULL},
    {0.05,   32*1024,  1.0, 1.0, RNG_LogNormal, 1234ULL},
    {0.05,    8*1024,  1.0, 1.0, RNG_LogNormal, 1234ULL},
    {0.0001, 32*1024, -1.0, 1.0,   RNG_Uniform, 1234ULL},
    {0.0006,  8*1024, -1.0, 1.0,   RNG_Uniform, 1234ULL},
    {0.005,  32*1024,  1.0, 1.0,    RNG_Gumbel, 1234ULL},
    {0.01,    8*1024,  1.0, 1.0,    RNG_Gumbel, 1234ULL},
    {0.005,  32*1024,  1.0, 1.0,  RNG_Logistic, 1234ULL},
    {0.01,    8*1024,  1.0, 1.0,  RNG_Logistic, 1234ULL},
    {0.005,  32*1024,  1.0, 1.0,       RNG_Exp, 1234ULL},
    {0.02,    8*1024,  1.0, 1.0,       RNG_Exp, 1234ULL},
    {0.005,  32*1024,  1.0, 1.0,  RNG_Rayleigh, 1234ULL},
    {0.02,    8*1024,  1.0, 1.0,  RNG_Rayleigh, 1234ULL},
    {0.005,  32*1024,  1.0, 1.0,   RNG_Laplace, 1234ULL},
    {0.02,    8*1024,  1.0, 1.0,   RNG_Laplace, 1234ULL}
};
TEST_P(RngTestD, Result){
    double meanvar[2];
    getExpectedMeanVar(meanvar);
    ASSERT_TRUE(match(meanvar[0], h_stats[0], CompareApprox<double>(params.tolerance)));
    ASSERT_TRUE(match(meanvar[1], h_stats[1], CompareApprox<double>(params.tolerance)));
}
INSTANTIATE_TEST_CASE_P(RngTests, RngTestD, ::testing::ValuesIn(inputsd));

} // end namespace Random
} // end namespace MLCommon

#include <gtest/gtest.h>
#include "matrix/math.h"
#include "random/rng.h"
#include "test_utils.h"


namespace MLCommon {
namespace Matrix {

template <typename Type>
__global__ void nativePowerKernel(Type* in, Type* out, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < len) {
    	out[idx] = in[idx] * in[idx];
    }
}

template <typename Type>
void naivePower(Type* in, Type* out, int len) {
    static const int TPB = 64;
    int nblks = ceildiv(len, TPB);
    nativePowerKernel<Type><<<nblks,TPB>>>(in, out, len);
    CUDA_CHECK(cudaPeekAtLastError());
}

template <typename Type>
__global__ void nativeSqrtKernel(Type* in, Type* out, int len) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < len) {
    	out[idx] = sqrt(in[idx]);
    }
}

template <typename Type>
void naiveSqrt(Type* in, Type* out, int len) {
    static const int TPB = 64;
    int nblks = ceildiv(len, TPB);
    nativeSqrtKernel<Type><<<nblks,TPB>>>(in, out, len);
    CUDA_CHECK(cudaPeekAtLastError());
}

template <typename Type>
__global__ void nativeSignFlipKernel(Type *in, Type *out, int rowCount, int colCount) {

	int d_i = blockIdx.x * rowCount;
	int end = d_i + rowCount;

	if (blockIdx.x < colCount) {
		Type max = 0.0;
		int max_index = 0;
		for (int i = d_i; i < end; i++) {
			Type val = in[i];
			if (val < 0.0) {
				val = -val;
			}
			if (val > max) {
				max = val;
				max_index = i;
			}
		}


		for (int i = d_i; i < end; i++) {
			if (in[max_index] < 0.0) {
			    out[i] = -in[i];
			} else {
				out[i] = in[i];
			}
		}
	}

	__syncthreads();
}

template <typename Type>
void naiveSignFlip(Type *in, Type *out, int rowCount, int colCount) {
    nativeSignFlipKernel<Type><<<colCount,1>>>(in, out, rowCount, colCount);
    CUDA_CHECK(cudaPeekAtLastError());
}

template <typename T>
struct MathInputs {
    T tolerance;
    int n_row;
    int n_col;
    int len;
    unsigned long long int seed;
};

template <typename T>
::std::ostream& operator<<(::std::ostream& os, const MathInputs<T>& dims) {
    return os;
}

template <typename T>
class MathTest: public ::testing::TestWithParam<MathInputs<T> > {
protected:
    void SetUp() override {
        params = ::testing::TestWithParam<MathInputs<T>>::GetParam();
        Random::Rng<T> r(params.seed);
        int len = params.len;

        allocate(in_power, len);
        allocate(out_power_ref, len);
        allocate(in_sqrt, len);
        allocate(out_sqrt_ref, len);
        allocate(in_sign_flip, len);
        allocate(out_sign_flip_ref, len);

        allocate(in_ratio, 4);
        T in_ratio_h[4] = { 1.0, 2.0, 2.0, 3.0 };
        updateDevice(in_ratio, in_ratio_h, 4);

        allocate(out_ratio_ref, 4);
        T out_ratio_ref_h[4] = { 0.125, 0.25, 0.25, 0.375 };
        updateDevice(out_ratio_ref, out_ratio_ref_h, 4);

        r.uniform(in_power, len, T(-1.0), T(1.0));
        r.uniform(in_sqrt, len, T(0.0), T(1.0));
        //r.uniform(in_ratio, len, T(0.0), T(1.0));
        r.uniform(in_sign_flip, len, T(0.0), T(1.0));

        naivePower(in_power, out_power_ref, len);
        power(in_power, len);

        naiveSqrt(in_sqrt, out_sqrt_ref, len);
        seqRoot(in_sqrt, len);

        ratio(in_ratio, in_ratio, 4);

        naiveSignFlip(in_sign_flip, out_sign_flip_ref, params.n_row, params.n_col);
        //signFlip(in_sign_flip, params.n_row, params.n_col);

    }

    void TearDown() override {
        CUDA_CHECK(cudaFree(in_power));
        CUDA_CHECK(cudaFree(out_power_ref));
        CUDA_CHECK(cudaFree(in_sqrt));
        CUDA_CHECK(cudaFree(out_sqrt_ref));
        CUDA_CHECK(cudaFree(in_ratio));
        CUDA_CHECK(cudaFree(out_ratio_ref));
        CUDA_CHECK(cudaFree(in_sign_flip));
        CUDA_CHECK(cudaFree(out_sign_flip_ref));
    }

protected:
    MathInputs<T> params;
    T *in_power, *out_power_ref, *in_sqrt, *out_sqrt_ref, *in_ratio, *out_ratio_ref, *in_sign_flip, *out_sign_flip_ref;
};

const std::vector<MathInputs<float> > inputsf = {
    {0.00001f, 1024, 1024, 1024*1024, 1234ULL}
};

const std::vector<MathInputs<double> > inputsd = {
    {0.00001, 1024, 1024, 1024*1024, 1234ULL}
};

typedef MathTest<float> MathPowerTestF;
TEST_P(MathPowerTestF, Result) {
    ASSERT_TRUE(devArrMatch(in_power, out_power_ref, params.len,
                            CompareApprox<float>(params.tolerance)));
}

typedef MathTest<double> MathPowerTestD;
TEST_P(MathPowerTestD, Result){
    ASSERT_TRUE(devArrMatch(in_power, out_power_ref, params.len,
                            CompareApprox<double>(params.tolerance)));
}

typedef MathTest<float> MathSqrtTestF;
TEST_P(MathSqrtTestF, Result) {
    ASSERT_TRUE(devArrMatch(in_sqrt, out_sqrt_ref, params.len,
                            CompareApprox<float>(params.tolerance)));
}

typedef MathTest<double> MathSqrtTestD;
TEST_P(MathSqrtTestD, Result){
    ASSERT_TRUE(devArrMatch(in_sqrt, out_sqrt_ref, params.len,
                            CompareApprox<double>(params.tolerance)));
}

typedef MathTest<float> MathRatioTestF;
TEST_P(MathRatioTestF, Result) {
    ASSERT_TRUE(devArrMatch(in_ratio, out_ratio_ref, 4,
                            CompareApprox<float>(params.tolerance)));
}

typedef MathTest<double> MathRatioTestD;
TEST_P(MathRatioTestD, Result){
    ASSERT_TRUE(devArrMatch(in_ratio, out_ratio_ref, 4,
                            CompareApprox<double>(params.tolerance)));
}

typedef MathTest<float> MathSignFlipTestF;
TEST_P(MathSignFlipTestF, Result) {
    ASSERT_TRUE(devArrMatch(in_sign_flip, out_sign_flip_ref, params.len,
                            CompareApprox<float>(params.tolerance)));
}

typedef MathTest<double> MathSignFlipTestD;
TEST_P(MathSignFlipTestD, Result){
    ASSERT_TRUE(devArrMatch(in_sign_flip, out_sign_flip_ref, params.len,
                            CompareApprox<double>(params.tolerance)));
}

INSTANTIATE_TEST_CASE_P(MathTests, MathPowerTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(MathTests, MathPowerTestD, ::testing::ValuesIn(inputsd));


INSTANTIATE_TEST_CASE_P(MathTests, MathSqrtTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(MathTests, MathSqrtTestD, ::testing::ValuesIn(inputsd));

INSTANTIATE_TEST_CASE_P(MathTests, MathRatioTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(MathTests, MathRatioTestD, ::testing::ValuesIn(inputsd));

INSTANTIATE_TEST_CASE_P(MathTests, MathSignFlipTestF, ::testing::ValuesIn(inputsf));

INSTANTIATE_TEST_CASE_P(MathTests, MathSignFlipTestD, ::testing::ValuesIn(inputsd));


} // end namespace LinAlg
} // end namespace MLCommon

#include <random/rng.h>
#include <linalg/cublas_wrappers.h>
#include <ml_utils.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/permutation_iterator.h>

#define IDX2C(i,j,ld) (j*ld + i)

using namespace MLCommon::LinAlg;
using namespace MLCommon;
using namespace ML::HMM;

namespace MLCommon {
namespace HMM {

template <typename T>
struct Inv_functor
{
        __host__ __device__
        T operator()(T& x)
        {
                return (T) 1.0 / x;
        }
};

template <typename T>
void gen_array(T* array, const int dim, paramsRandom paramsRd){
        MLCommon::Random::Rng<T> rng(paramsRd.seed);
        rng.uniform(array, dim, paramsRd.start, paramsRd.end);
}

template <typename T>
void normalize_matrix(T* array, int n_rows, int n_cols){
        // cublas handles
        cublasHandle_t cublas_handle;
        cublasCreate(&cublas_handle);

        // initializations
        T *sums, *ones;
        cudaMalloc(&sums, n_rows * sizeof(T));
        cudaMalloc(&ones, n_cols * sizeof(T));

        thrust::device_ptr<T> sums_th(sums);
        thrust::device_ptr<T> ones_th(ones);

        const T alpha = (T) 1;
        const T beta = (T) 0;

        thrust::fill(sums_th, sums_th + n_rows, beta);
        thrust::fill(ones_th, ones_th + n_cols, alpha);

        // Compute the sum of each row

        CUBLAS_CHECK(cublasgemv(cublas_handle, CUBLAS_OP_N, n_rows, n_cols, &alpha,
                                array, n_rows, ones, 1, &beta, sums, 1));

        // Inverse the sums
        thrust::transform(sums_th, sums_th + n_rows, sums_th, Inv_functor<T>());

        // Multiply by the inverse
        CUBLAS_CHECK(cublasdgmm(cublas_handle, CUBLAS_SIDE_LEFT, n_rows, n_cols, array,
                                n_rows, sums, 1, array, n_rows));
}

template <typename T>
void gen_trans_matrix(T* matrix, int n_rows, int n_cols, T start, T end,
                      unsigned long long seed){
        int dim = n_rows * n_cols;
        generate_array(matrix, dim, start, end, seed);
        normalize_matrix(matrix, n_rows, n_cols);
}

}
}

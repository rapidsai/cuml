#include <random/rng.h>
#include <linalg/cublas_wrappers.h>

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

template <typename T>
void print_matrix(T* gpu, int rows, int cols, const std::string& msg){
    T* cpu;
    cpu = (T *)malloc(sizeof(T)*rows*cols);
    updateHost(cpu, gpu, rows*cols);
    printf("\n\n");
    printf("%s\n", msg.c_str());
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++)
            printf("%f | ", cpu[IDX2C(i, j , rows)]);
        printf("\n");
    }
}

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
void generate_random_array(T* random_array, const int array_size, T start, T end, unsigned long long _s){
    MLCommon::Random::Rng<T> rng(_s);
    rng.uniform(random_array, array_size, start, end);
}

template <typename T>
void normalize_matrix(T* random_array, int n_rows, int n_cols){
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
    	           random_array, n_rows, ones, 1, &beta, sums, 1));

    // Inverse the sums
    thrust::transform(sums_th, sums_th + n_rows, sums_th, Inv_functor<T>());

    print_matrix(random_array, n_rows, n_cols, "random array");
    print_matrix(sums, n_rows, 1, "inv sums array");

    // Multiply by the inverse
    CUBLAS_CHECK(cublasdgmm(cublas_handle, CUBLAS_SIDE_LEFT, n_rows, n_cols, random_array,
                n_rows, sums, 1, random_array, n_rows));
}

int main(){
    float *d_random_array;
    const int n_rows =  3;
    const int n_cols =  4;
    const int array_size = n_rows * n_cols;
    unsigned long long _s = 1234ULL;

    float start = 0;
    float end = 100;

    cudaMalloc(&d_random_array, array_size * sizeof(float));

    generate_random_array(d_random_array, array_size, start, end, _s);
    normalize_matrix(d_random_array, n_rows, n_cols);

    print_matrix(d_random_array, n_rows, n_cols, "random array");

    return 0;
}

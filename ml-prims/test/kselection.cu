#include "selection/kselection.h"
#include "random/rng.h"
#include <gtest/gtest.h>
#include <limits>

namespace MLCommon {
namespace Selection {

template <typename TypeV, typename TypeK, int N, int TPB, bool Greater>
__global__ void sortTestKernel(TypeK* key) {
    KVArray<TypeV,TypeK,N,Greater> arr;
    #pragma unroll
    for(int i=0;i<N;++i) {
        arr.arr[i].val = (TypeV)laneId();
        arr.arr[i].key = (TypeK)laneId();
    }
    warpFence();
    arr.sort();
    warpFence();
    #pragma unroll
    for(int i=0;i<N;++i)
        arr.arr[i].store(nullptr, key+threadIdx.x+i*TPB);
}

template <typename TypeV, typename TypeK, int N, int TPB, bool Greater>
void sortTest(TypeK* key) {
    TypeK* dkey;
    CUDA_CHECK(cudaMalloc((void**)&dkey, sizeof(TypeK)*TPB*N));
    sortTestKernel<TypeV,TypeK,N,TPB,Greater><<<1,TPB>>>(dkey);
    CUDA_CHECK(cudaPeekAtLastError());
    updateHost<TypeK>(key, dkey, TPB*N);
    CUDA_CHECK(cudaFree(dkey));
}

TEST(KVArray, Sort) {
    static const int N = 4;
    int* key = new int[N*WarpSize];
    sortTest<float,int,N,WarpSize,true>(key);
    // for(int i=0;i<N*WarpSize;++i) {
    //     printf("%d,", key[i]);
    // }
    // printf("\n");
    for(int i=0;i<N*WarpSize;++i) {
        ASSERT_EQ(i/N, key[i]);
    }
    delete [] key;
}


TEST(WarpTopK, Test) {
    static const int rows = 1;
    static const int cols = 256;
    static const int k = 32;
    float* arr;
    allocate(arr, rows*cols);
    float* outv;
    int* outk;
    allocate(outk, rows*k);
    allocate(outv, rows*k);
    Random::Rng<float> r(1234ULL);
    r.uniform(arr, rows*cols, -1.f, 1.f);
    warpTopK<float,int,true,false>(outv, outk, arr, k, rows, cols);
    float* h_outv = new float[rows*k];
    updateHost(h_outv, outv, rows*k);
    int* h_outk = new int[rows*k];
    updateHost(h_outk, outk, rows*k);
    for(int i=0;i<rows;++i) {
        printf("%d", i);
        for(int j=0;j<k;++j) {
            printf(",%f:%d", h_outv[i*k+j], h_outk[i*k+j]);
        }
        printf("\n");
    }
    delete [] h_outv;
    delete [] h_outk;
    float* h_arr = new float[rows*cols];
    updateHost(h_arr, arr, rows*cols);
    for(int i=0;i<rows;++i) {
        printf("%d", i);
        for(int j=0;j<cols;++j) {
            printf(",%f", h_arr[i*cols+j]);
        }
        printf("\n");
    }
    delete [] h_arr;
    CUDA_CHECK(cudaFree(outv));
    CUDA_CHECK(cudaFree(outk));
    CUDA_CHECK(cudaFree(arr));
}

} // end namespace Selection
} // end namespace MLCommon

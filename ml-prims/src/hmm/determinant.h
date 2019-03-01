#pragma once

#include <cublas_v2.h>

#include "utils.h"
#include "cuda_utils.h"
#include "linalg/cublas_wrappers.h"
#include "linalg/cusolver_wrappers.h"


#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace MLCommon {

template <typename T>
struct StrideFunctor
{
        int stride;
        T* array;
        StrideFunctor(int _stride, T* _array){
                stride = _stride;
                array = _array;
        }

        __host__ __device__
        T operator() (int idx)
        {
                return *(array + idx * stride);
        }
};

template <typename T>
T diag_product(T* array, int n, int ldda){
        thrust::counting_iterator<int> first(0);
        thrust::counting_iterator<int> last = first + n;
        StrideFunctor<T> strideFn(ldda + 1, array);
        return thrust::reduce(thrust::make_transform_iterator(first, strideFn),
                              thrust::make_transform_iterator(last, strideFn),
                              (T) 1., thrust::multiplies<T>());
}

template <typename T>
struct Determinant
{
        int n, ldda;
        T *tempM;
        bool is_hermitian;

        int *devIpiv;

        int *info, info_h;
        cusolverDnHandle_t *handle;
        cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
        int WsSize;
        T *Ws = NULL;

        Determinant(int _n, int _ldda, cusolverDnHandle_t *_handle,
                    bool _is_hermitian){
                n = _n;
                ldda = _ldda;
                is_hermitian = _is_hermitian;
                handle = _handle;


                if (is_hermitian) {
                        CUSOLVER_CHECK(LinAlg::cusolverDnpotrf_bufferSize(*handle, uplo, n, tempM, ldda, &WsSize));

                }
                else
                {
                        CUSOLVER_CHECK(LinAlg::cusolverDngetrf_bufferSize(*handle,  n, n, tempM, ldda, &WsSize));
                        allocate(devIpiv, n);
                }

                allocate(Ws, WsSize);
                allocate(info, 1);
                allocate(tempM, ldda * n);
        }

        T compute(T* M){
                copy(tempM, M, ldda * n);

                if(is_hermitian) {
                        CUSOLVER_CHECK(LinAlg::cusolverDnpotrf(*handle,
                                                               uplo,
                                                               n,
                                                               tempM,
                                                               ldda,
                                                               Ws,
                                                               WsSize,
                                                               info));
                }
                else{
                        CUSOLVER_CHECK(LinAlg::cusolverDngetrf(*handle,
                                                               n,
                                                               n,
                                                               tempM,
                                                               ldda,
                                                               Ws,
                                                               devIpiv,
                                                               info));

                }
                updateHost(&info_h, info, 1);
                ASSERT(info_h == 0,
                       "sigma: error in determinant, info=%d | expected=0", info_h);

                T prod = diag_product(tempM, n, ldda);
                return prod;
        }

        void TearDown() {
                CUDA_CHECK(cudaFree(Ws));
                CUDA_CHECK(cudaFree(tempM));
                CUDA_CHECK(cudaFree(info));

                if (is_hermitian) {
                        CUDA_CHECK(cudaFree(devIpiv));
                }
        }

};

}

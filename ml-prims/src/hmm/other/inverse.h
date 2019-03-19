#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>
#include <cublas_v2.h>

#include "utils.h"
#include "cuda_utils.h"
#include "linalg/cublas_wrappers.h"
#include "linalg/cusolver_wrappers.h"

using namespace MLCommon::LinAlg;
using namespace MLCommon;

namespace MLCommon {

template <typename T>
struct Inverse {
        int nDim;
        T *tempM;
        int *piv;

        // Cusolver variables
        int *info, info_h;
        cusolverDnHandle_t *cusolverHandle;

        int wsSize;
        T *workspace_lu = nullptr;

        Inverse(int _nDim, cusolverDnHandle_t *_cusolverHandle){
                nDim = _nDim;
                cusolverHandle = _cusolverHandle;
                CUSOLVER_CHECK(cusolverDngetrf_bufferSize(*cusolverHandle, nDim, nDim, tempM, nDim, &wsSize));

                allocate(tempM, nDim * nDim);
                allocate(workspace_lu, wsSize);
                allocate(piv, nDim);
                allocate(info, 1);
        }


        void compute(T* M, T* invM){
                copy(tempM, M, nDim * nDim);

                // We assume M is hermitian
                CUSOLVER_CHECK(cusolverDngetrf(*cusolverHandle, nDim, nDim, tempM, nDim, workspace_lu, piv, info));
                updateHost(&info_h, info, 1);
                ASSERT(info_h == 0, "mlcommon::hmm::inverse: LU decomp, info returned val=%d", info_h);

                // Making ID matrix
                make_ID_matrix(invM, nDim);

                // TODO : Investigate the cuda-memcheck error at this stage
                CUSOLVER_CHECK(cusolverDngetrs(*cusolverHandle, CUBLAS_OP_N, nDim, nDim, tempM, nDim, piv, invM, nDim, info));

                updateHost(&info_h, info, 1);
                ASSERT(info_h == 0,
                       "mlcommon::hmm::inverse "
                       "returned val=%d", info_h);
        }

        void TearDown() {
                CUDA_CHECK(cudaFree(tempM));
                CUDA_CHECK(cudaFree(workspace_lu));
                CUDA_CHECK(cudaFree(piv));
                CUDA_CHECK(cudaFree(info));
        }
};

}

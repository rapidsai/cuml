/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "cuda_utils.h"
#include "pack.h"

namespace Dbscan {
namespace EpsNeigh {
namespace Algo1 {

/** number of D row elements loaded per main loop iteration */
static const int DBLOCK = 32;
/** number of N elements worked per CTA */
static const int NBLOCK = 32;
/** number of blocks worked per CTA */
static const int NBLKS = 8;
/** num threads per CTA */
static const int NUM_THREADS = NBLOCK * NBLKS;
/** number of elements loaded per main loop iteration */
static const int NUM = NUM_THREADS / NBLOCK;
/** number of elements per thread */
static const int NUMT = NBLOCK / NUM;

template <int _nThreads, int _dBlock>
struct KernelPolicy {
    enum {
        /** number of threads per CTA */
        NTHREADS = _nThreads,

        /** num D elements in the set of LDGs per main loop iteration */
        DBLOCK = _dBlock,

        /** number of N elements worked per CTA (always 32!) */
        NBLOCK = 32,

        /** number of elements loaded per main loop iteration */
        NUM = NTHREADS / NBLOCK,
        /** number of elements per thread */
        NUMT = NBLOCK / NUM,

        /** stride for accessing shared mem */
        SMEM_COL = NBLOCK + 1,
        /** size of an smem page */
        SMEM_PAGE = SMEM_COL * DBLOCK
    };
};

struct P256: KernelPolicy<256, 32> {};
struct P128: KernelPolicy<128, 32> {};


template <typename Type, typename Policy>
struct EpsNeigh {
    DI EpsNeigh(char* _adj, Type* _x, int _N, int _D, Type _eps, Type* _smem) {
        N = _N;
        D = _D;
        eps2 = _eps * _eps;
        x = _x;
        adj = _adj;
        sxptr = _smem;
        syptr = _smem + Policy::SMEM_PAGE;
    }

    DI void run() {
        initProlog();
        for(int d=0;d<D;d+=Policy::DBLOCK) {
            loadFromGmem(d);
            __syncthreads(); // make sure that prev-iter has used all data!
            storeToSmem();
            __syncthreads();
            for(int i=0;i<DBLOCK;++i) {
                loadFromSmem(i);
                accumulate();
            }
        }
        initEpilog();
        epilog();
    }

    DI void initProlog() {
        warpId = threadIdx.x / Policy::NBLOCK;
        laneId = threadIdx.x % Policy::NBLOCK;
        for(int i=0;i<Policy::NUMT;++i) {
            rid[i] = (blockIdx.y * Policy::NBLOCK) + warpId + (i * Policy::NUM);
        }
        for(int i=0;i<Policy::NUMT;++i) {
            rvalid[i] = rid[i] < N;
        }
        for(int i=0;i<Policy::NUMT;++i) {
            cid[i] = (blockIdx.x * Policy::NBLOCK) + warpId + (i * Policy::NUM);
        }
        for(int i=0;i<Policy::NUMT;++i) {
            cvalid[i] = cid[i] < N;
        }
        for(int i=0;i<Policy::NUMT;++i) {
            sum[i] = Zero;
        }
        sxWr = sxptr + (laneId * Policy::SMEM_COL) + warpId;
        syWr = syptr + (laneId * Policy::SMEM_COL) + warpId;
        sxRd = sxptr + warpId;
        syRd = syptr + laneId;
    }

    DI void loadFromGmem(int d) {
        for(int i=0;i<Policy::NUMT;++i) {
            xval[i] = (rvalid[i] && (d+laneId < D))?
                __ldg(x+rid[i]*D+d+laneId) : Zero;
            yval[i] = ((d+laneId < D) && cvalid[i])?
                __ldg(x+cid[i]*D+d+laneId) : Zero;
        }
    }

    DI void storeToSmem() {
        for(int i=0;i<Policy::NUMT;++i) {
            sxWr[i*Policy::NUM] = xval[i];
            syWr[i*Policy::NUM] = yval[i];
        }
    }

    DI void loadFromSmem(int i) {
        for(int j=0;j<Policy::NUMT;++j) {
            xdata[j] = sxRd[(i * Policy::SMEM_COL) + (j * Policy::NUM)];
        }
        ydata = syRd[i * Policy::SMEM_COL];
    }

    DI void accumulate() {
        for(int j=0;j<Policy::NUMT;++j) {
            diff[j] = xdata[j] - ydata;
        }
        for(int j=0;j<Policy::NUMT;++j) {
            sum[j] += (diff[j] * diff[j]);
        }
    }

    DI void initEpilog() {
        for(int i=0;i<Policy::NUMT;++i) {
            wrRid[i] = (blockIdx.y * Policy::NBLOCK) + warpId + (i * Policy::NUM);
        }
        wrCid = (blockIdx.x * Policy::NBLOCK) + laneId;
        for(int i=0;i<Policy::NUMT;++i) {
            wrRvalid[i] = wrRid[i] < N;
        }
        wrCvalid = wrCid < N;
    }

    DI void epilog() {
        for(int i=0;i<Policy::NUMT;++i) {
            if(wrRvalid[i] && wrCvalid) {
                adj[wrRid[i]*N+wrCid] = (sum[i] <= eps2);
            }
        }
    }

    static const Type Zero = (Type)0;

    int N, D;
    Type eps2;
    Type* x;
    char* adj;

    Type *sxptr, *syptr, *sxWr, *syWr, *sxRd, *syRd;
    int warpId, laneId;
    int rid[Policy::NUMT], cid[Policy::NUMT];
    bool rvalid[Policy::NUMT], cvalid[Policy::NUMT];
    Type sum[Policy::NUMT];
    Type xval[Policy::NUMT], yval[Policy::NUMT];
    Type xdata[Policy::NUMT], ydata, diff[Policy::NUMT];
    int wrRid[Policy::NUMT], wrCid;
    bool wrRvalid[Policy::NUMT], wrCvalid;
};


/**
 * @brief Naive distance matrix evaluation and epsilon neighborhood construction
 * @param adj eps-neighborhood (aka adjacent matrix)
 * @param x the input buffer
 * @param N number of rows
 * @param D number of columns
 * @note Algo1 tries to improve naive implementation by having coalesced global
 *  mem as well as conflict-free shared mem accesses.
 */
template <typename Type, typename Policy>
__global__ void eps_neigh_kernel(Pack<Type> data) {
    __shared__ Type smem[Policy::SMEM_PAGE*2];
    EpsNeigh<Type,Policy>
        (data.adj, data.x, data.N, data.D, data.eps, smem).run();
}

template <typename Type, typename Policy>
void launcher(Pack<Type> data, cudaStream_t stream) {
    dim3 grid(ceildiv(data.N, Policy::NBLOCK), ceildiv(data.N, Policy::NBLOCK), 1);
    dim3 blk(Policy::NTHREADS, 1, 1);
    void *ptr = (void*)&data;
    void *func = (void*)eps_neigh_kernel<Type, Policy>;
    cudaLaunchKernel(func, grid, blk, &ptr, 0, stream);
    CUDA_CHECK(cudaPeekAtLastError());
}

} // namespace Algo1
} // namespace EpsNeigh
} // namespace Dbscan

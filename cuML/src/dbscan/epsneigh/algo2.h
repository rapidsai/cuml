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

#include "../cuda_utils.h"
#include "../types.h"
#include "pack.h"

namespace Dbscan {
namespace EpsNeigh {
namespace Algo2 {


template <typename Type, int _nThreads, int _dBlock, int _rBlock, int _cBlock>
struct KernelPolicy {
    enum {
        /** number of threads per CTA */
        NTHREADS = _nThreads,

        /**
         * num D elements in the set of LDGs per main loop iteration.
         * Must be:
         *  . Po2
         *  . divisible by 4
         *  . <= 32
         *  . thus, one of {4, 8, 16, 32}
         */
        DBLOCK = _dBlock,

        /**
         * number of N row elements worked per CTA
         * Must be:
         *  . po2
         */
        RBLOCK = _rBlock,

        /**
         * number of N col elements worked per CTA
         * Must be:
         *  . po2
         */
        CBLOCK = _cBlock,

        /** number of rows loaded per LDG/LDS */
        RPL = NTHREADS / DBLOCK,
        /** number of rows per thread */
        RPT = RBLOCK / RPL,

        /** number of cols loaded per LDS/LDS */
        CPL = NTHREADS / DBLOCK,
        /** number of cols per thread */
        CPT = CBLOCK / CPL,

        /** size (in B) of one input element */
        INSIZE = sizeof(Type),

        /**
         * stride for accessing shared mem
         * @todo: un-hardcode the '+4' below to avoid bank conflicts
         */
        SMEM_COL = DBLOCK + 4,
        /** column size in bytes */
        SMEM_COL_B = SMEM_COL * INSIZE,
        /** size of an smem X page */
        SMEM_XPAGE = SMEM_COL_B * RBLOCK,
        /** size of an smem Y page */
        SMEM_YPAGE = SMEM_COL_B * CBLOCK,
        /** size of smem */
        SMEM_PAGE = SMEM_XPAGE + SMEM_YPAGE
    };
};

template <typename Type>
struct P64x64: KernelPolicy<Type, 256, 16, 64, 64> {
};

///@todo: below 2 policies cause spills on sm_35 arch!
template <typename Type>
struct P128x32: KernelPolicy<Type, 256, 16, 128, 32> {
};

template <typename Type>
struct P32x128: KernelPolicy<Type, 256, 16, 32, 128> {
};

///@todo: all the below policies currently cause register spills!
template <typename Type>
struct P128x64: KernelPolicy<Type, 256, 16, 128, 64> {
};

template <typename Type>
struct P64x128: KernelPolicy<Type, 256, 16, 64, 128> {
};

template <typename Type>
struct P128x128: KernelPolicy<Type, 256, 16, 128, 128> {
};


template <typename Type, typename Policy> 
struct EpsNeigh {
    DI EpsNeigh(char* _adj, Type* _x, int _N, int _D, Type _eps, uint32_t _smem) {
        N = _N;
        D = _D;
        eps2 = _eps * _eps;
        x = _x;
        adj = _adj;
        sxptr = _smem;
        syptr = _smem + Policy::SMEM_XPAGE;
    }

    DI void run() {
        initProlog();
        for(int d=0;d<D;d+=Policy::DBLOCK) {
            loadFromGmem(d);
            storeToSmem();
            __syncthreads();
            loadFromSmem<0>(0);
            for(int i=0;i<Policy::DBLOCK;i+=4) {
                int page = (i / 4) % 2;
                if(i+4 < Policy::DBLOCK) {
                    if(page^1) {
                        loadFromSmem<1>(i+4);
                    } else {
                        loadFromSmem<0>(i+4);
                    }
                }
                if(page) {
                    accumulate<1>();
                } else {
                    accumulate<0>();
                }
            }
            if(sxRd < Policy::SMEM_XPAGE) {
                sxRd += Policy::SMEM_PAGE;
                syRd += Policy::SMEM_PAGE;
            } else {
                sxRd -= Policy::SMEM_PAGE;
                syRd -= Policy::SMEM_PAGE;
            }
        }
        initEpilog();
        epilog();
    }

    DI void initProlog() {
        xRowLdg = threadIdx.x / Policy::RPL;
        colId = threadIdx.x % Policy::DBLOCK;
        yRowLdg = threadIdx.x / Policy::CPL;
        for(int i=0;i<Policy::RPT;++i) {
            rid[i] = (blockIdx.y * Policy::RBLOCK) + (Policy::RPL * i) + xRowLdg;
        }
        for(int i=0;i<Policy::CPT;++i) {
            cid[i] = (blockIdx.x * Policy::CBLOCK) + (Policy::CPL * i) + yRowLdg;
        }
        for(int i=0;i<Policy::RPT;++i) {
            rvalid[i] = rid[i] < N;
        }
        for(int i=0;i<Policy::CPT;++i) {
            cvalid[i] = cid[i] < N;
        }
        sxWr = sxptr + ((colId + (xRowLdg * Policy::SMEM_COL)) * Policy::INSIZE);
        syWr = syptr + ((colId + (yRowLdg * Policy::SMEM_COL)) * Policy::INSIZE);
        rowLds = threadIdx.x / Policy::CPL;
        colLds = threadIdx.x % Policy::CPL;
        sxRd = sxptr + (rowLds * Policy::SMEM_COL_B);
        syRd = syptr + (colLds * Policy::SMEM_COL_B);
        for(int i=0;i<Policy::RPT;++i) {
            for(int j=0;j<Policy::CPT;++j) {
                sum[i][j] = Zero;
            }
        }
    }

    DI void loadFromGmem(int d) {
        for(int i=0;i<Policy::RPT;++i) {
            xval[i] = (rvalid[i] && (d+colId < D))?
                __ldg(x+rid[i]*D+d+colId) : Zero;
        }
        for(int i=0;i<Policy::CPT;++i) {
            yval[i] = ((d+colId < D) && cvalid[i])?
                __ldg(x+cid[i]*D+d+colId) : Zero;
        }
    }

    DI void storeToSmem() {
        for(int i=0;i<Policy::RPT;++i) {
            sts(sxWr + (Policy::RPL * i * Policy::SMEM_COL_B), xval[i]);
        }
        for(int i=0;i<Policy::CPT;++i) {
            sts(syWr + (Policy::CPL * i * Policy::SMEM_COL_B), yval[i]);
        }
        if(sxWr < Policy::SMEM_XPAGE) {
            sxWr += Policy::SMEM_PAGE;
            syWr += Policy::SMEM_PAGE;
        } else {
            sxWr -= Policy::SMEM_PAGE;
            syWr -= Policy::SMEM_PAGE;
        }
    }

    template <int PAGE>
    DI void loadFromSmem(int i) {
        for(int j=0;j<Policy::RPT;++j) {
            lds(xdata[PAGE][j],
                (sxRd + ((i + (Policy::RPL*j*Policy::SMEM_COL)) * Policy::INSIZE)));
        }
        for(int j=0;j<Policy::CPT;++j) {
            lds(ydata[PAGE][j],
                (syRd + ((i + (Policy::CPL*j*Policy::SMEM_COL)) * Policy::INSIZE)));
        }
    }

    template <int PAGE>
    DI void accumulate() {
        for(int j=0;j<Policy::RPT;++j) {
            for(int k=0;k<Policy::CPT;++k) {
                Type4<Type> diff = xdata[PAGE][j] - ydata[PAGE][k];
                sum[j][k] += (diff.a * diff.a);
                sum[j][k] += (diff.b * diff.b);
                sum[j][k] += (diff.c * diff.c);
                sum[j][k] += (diff.d * diff.d);
            }
        }
    }

    DI void initEpilog() {
        for(int i=0;i<Policy::RPT;++i) {
            rid[i] = (blockIdx.y * Policy::RBLOCK) + (Policy::RPL * i) + rowLds;
        }
        for(int i=0;i<Policy::CPT;++i) {
            cid[i] = (blockIdx.x * Policy::CBLOCK) + (Policy::CPL * i) + colLds;
        }
        for(int i=0;i<Policy::RPT;++i) {
            rvalid[i] = rid[i] < N;
        }
        for(int i=0;i<Policy::CPT;++i) {
            cvalid[i] = cid[i] < N;
        }
    }

    DI void epilog() {
        for(int i=0;i<Policy::RPT;++i) {
            for(int j=0;j<Policy::CPT;++j) {
                if(rvalid[i] && cvalid[j]) {
                    adj[rid[i]*N+cid[j]] = (sum[i][j] <= eps2);
                }
            }
        }
    }

    static const Type Zero = (Type)0;

    int N, D;
    Type eps2;
    Type *x;
    char *adj;

    uint32_t sxptr, syptr;
    uint32_t sxWr, sxRd, syWr, syRd;
    int rid[Policy::RPT], cid[Policy::CPT];
    bool rvalid[Policy::RPT], cvalid[Policy::CPT];
    int xRowLdg, yRowLdg, colId;
    int rowLds, colLds;
    Type sum[Policy::RPT][Policy::CPT];
    Type xval[Policy::RPT], yval[Policy::CPT];
    Type4<Type> xdata[2][Policy::RPT], ydata[2][Policy::CPT];
};


/**
 * @brief Distance matrix evaluation and epsilon neighborhood construction
 * @param adj eps-neighborhood (aka adjacent matrix)
 * @param x the input buffer
 * @param N number of rows
 * @param D number of columns
 * @note Algo2 tries to improve naive implementation by having:
 *   . multiple row/col elements per thread
 *   . paged loads from smem and FMA's to improve utilization
 */
template <typename Type, typename Policy>
__global__ __launch_bounds__(Policy::NTHREADS, 2)
    void eps_neigh_kernel(Pack<Type> data) {
    // HACK HACK HACK! dynamic smem start address assumed to be zero!!!
    EpsNeigh<Type,Policy>(data.adj, data.x, data.N, data.D, data.eps, 0).run();
}

template <typename Type, typename Policy>
void launcher(Pack<Type> data, cudaStream_t stream) {
    dim3 grid(ceildiv(data.N, Policy::CBLOCK), ceildiv(data.N, Policy::RBLOCK), 1);
    dim3 blk(Policy::NTHREADS, 1, 1);
    void *ptr = (void*)&data;
    void *func = (void*)eps_neigh_kernel<Type, Policy>;
    cudaLaunchKernel(func, grid, blk, &ptr, Policy::SMEM_PAGE*2, stream);
    CUDA_CHECK(cudaPeekAtLastError());
}

} // namespace Algo2
} // namespace EpsNeigh
} // namespace Dbscan

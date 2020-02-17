/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cuda_utils.h>
#include <stdint.h>
#include <cub/cub.cuh>
#include <limits>

#if (ENABLE_MEMCPY_ASYNC == 1)
#include <cuda_pipeline.h>
#endif

namespace MLCommon {
namespace Distance {

DI void sts(float* addr, const float& x) { *addr = x; }
DI void sts(float* addr, const float (&x)[1]) { *addr = x[0]; }
DI void sts(float* addr, const float (&x)[2]) {
  float2 v2 = make_float2(x[0], x[1]);
  auto* s2 = reinterpret_cast<float2*>(addr);
  *s2 = v2;
}
DI void sts(float* addr, const float (&x)[4]) {
  float4 v4 = make_float4(x[0], x[1], x[2], x[3]);
  auto* s4 = reinterpret_cast<float4*>(addr);
  *s4 = v4;
}
DI void sts(double* addr, const double& x) { *addr = x; }
DI void sts(double* addr, const double (&x)[1]) { *addr = x[0]; }
DI void sts(double* addr, const double (&x)[2]) {
  double2 v2 = make_double2(x[0], x[1]);
  auto* s2 = reinterpret_cast<double2*>(addr);
  *s2 = v2;
}

DI void lds(float& x, float* addr) { x = *addr; }
DI void lds(float (&x)[1], float* addr) { x[0] = *addr; }
DI void lds(float (&x)[2], float* addr) {
  auto* s2 = reinterpret_cast<float2*>(addr);
  auto v2 = *s2;
  x[0] = v2.x;
  x[1] = v2.y;
}
DI void lds(float (&x)[4], float* addr) {
  auto* s4 = reinterpret_cast<float4*>(addr);
  auto v4 = *s4;
  x[0] = v4.x;
  x[1] = v4.y;
  x[2] = v4.z;
  x[3] = v4.w;
}
DI void lds(double& x, double* addr) { x = *addr; }
DI void lds(double (&x)[1], double* addr) { x[0] = *addr; }
DI void lds(double (&x)[2], double* addr) {
  auto* s2 = reinterpret_cast<double2*>(addr);
  auto v2 = *s2;
  x[0] = v2.x;
  x[1] = v2.y;
}

DI void ldg(float& x, float* addr) {
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(x) : "l"(addr));
}
DI void ldg(float (&x)[1], float* addr) {
  asm volatile("ld.global.cg.f32 %0, [%1];" : "=f"(x[0]) : "l"(addr));
}
DI void ldg(float (&x)[2], float* addr) {
  asm volatile("ld.global.cg.v2.f32 {%0, %1}, [%2];"
               : "=f"(x[0]), "=f"(x[1])
               : "l"(addr));
}
DI void ldg(float (&x)[4], float* addr) {
  asm volatile("ld.global.cg.v4.f32 {%0, %1, %2, %3}, [%4];"
               : "=f"(x[0]), "=f"(x[1]), "=f"(x[2]), "=f"(x[3])
               : "l"(addr));
}
DI void ldg(double& x, double* addr) {
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(x) : "l"(addr));
}
DI void ldg(double (&x)[1], double* addr) {
  asm volatile("ld.global.cg.f64 %0, [%1];" : "=d"(x[0]) : "l"(addr));
}
DI void ldg(double (&x)[2], double* addr) {
  asm volatile("ld.global.cg.v2.f64 {%0, %1}, [%2];"
               : "=d"(x[0]), "=d"(x[1])
               : "l"(addr));
}

template <typename LabelT, typename DataT>
struct KVPMinReduce {
  typedef cub::KeyValuePair<LabelT, DataT> KVP;

  DI KVP operator()(const KVP& a, const KVP& b) {
    return b.value < a.value ? b : a;
  }
};  // KVPMinReduce

template <typename LabelT, typename DataT>
struct MinAndDistanceReduceOp {
  typedef typename cub::KeyValuePair<LabelT, DataT> KVP;
  DI void operator()(KVP* out, const KVP& other) {
    if (other.value < out->value) {
      out->key = other.key;
      out->value = other.value;
    }
  }

  DI void init(KVP* out, DataT maxVal) {
    out->key = -1;
    out->value = maxVal;
  }
};

template <typename LabelT, typename DataT>
struct MinReduceOp {
  typedef typename cub::KeyValuePair<LabelT, DataT> KVP;
  DI void operator()(DataT* out, const KVP& other) {
    if (other.value < *out) {
      *out = other.value;
    }
  }

  DI void init(DataT* out, DataT maxVal) { *out = maxVal; }
};

/**
 * @brief This is the central enum that should be used to configure the perf
 *        landscape of the fused kernel.
 * @tparam DataT the IO and math datatype
 * @tparam _veclen number of k-elements loaded by each thread for every LDG call
 *                 it makes. This should be configured based on the input 'k'
 *                 value and the input data type. For eg: if DataT = float and
 *                 k is multiples of 4, then setting this to 4 gives the best
 *                 LDG pattern. Possible values are {1, 2, 4}.
 * @tparam _kblk number of k-elements operated upon per main-loop iteration.
 *               Therefore total number of main-loop iterations will be
 *               ceil(k / _kblk). This must be multiples of `_veclen`. Do note
 *               that bigger this value, the greater shared mem requirement.
 * @tparam _rpt Defines the number of rows that a given thread accumulates on.
 *              This directly results in increased register pressure. This also
 *              is used to compute the number of m-elements worked upon by each
 *              thread block.
 * @tparam _rpt Defines the number of cols that a given thread accumulates on.
 *              This directly results in increased register pressure. This also
 *              is used to compute the number of n-elements worked upon by each
 *              thread block.
 * @tparam _tr Number of threads working on the same output column. This is used
 *             to compute the number of m-elements worked upon by each threadblk
 *             This also determines the number of threads per thread block
 * @tparam _tc Number of threads working on the same output row. This is used to
 *             compute the number of m-elements worked upon by each threadblk
 *             This also determines the number of threads per thread block
 */
template <typename DataT, int _veclen, int _kblk, int _rpt, int _cpt, int _tr,
          int _tc>
struct KernelPolicy {
  enum {
    /** number of elements along K worked upon per main loop iteration */
    Kblk = _kblk,
    /** number of elements loaded per LDG */
    Veclen = _veclen,
    /** number of rows a thread works on for accumulation */
    AccRowsPerTh = _rpt,
    /** number of cols a thread works on for accumulation */
    AccColsPerTh = _cpt,
    /** number of threads working the same output col */
    AccThRows = _tr,
    /** number of threads working the same output row */
    AccThCols = _tc,
    /** total threads per block */
    Nthreads = AccThRows * AccThCols,
    /** output tile size along rows */
    Mblk = AccRowsPerTh * AccThRows,
    /** output tile size along cols */
    Nblk = AccColsPerTh * AccThCols,
    /** number of threads loading a single row */
    LdgThK = Kblk / Veclen,
    /** number of LDGs issued by a single thread for X */
    LdgPerThX = Mblk * LdgThK / Nthreads,
    /** number of LDGs issued by a single thread for Y */
    LdgPerThY = Nblk * LdgThK / Nthreads,
    /** number of rows of X covered per LDG */
    LdgRowsX = Mblk / LdgPerThX,
    /** number of rows of Y covered per LDG */
    LdgRowsY = Nblk / LdgPerThY,
    /** stride for accessing X/Y data in shared mem */
    SmemStride = Kblk + Veclen,
    /** size of one page for storing X data */
    SmemPageX = SmemStride * Mblk,
    /** size of one page for storing Y data */
    SmemPageY = SmemStride * Nblk,
    /** size of one smem page */
    SmemPage = SmemPageX + SmemPageY,
    /** size (in B) for smem needed */
    SmemSize = 2 * SmemPage * sizeof(DataT),
  };  // enum
};    // struct KernelPolicy

template <typename DataT, int _veclen>
struct Policy4x4 {};

template <int _veclen>
struct Policy4x4<float, _veclen> {
  typedef KernelPolicy<float, _veclen, 32, 4, 4, 16, 16> Policy;
};

template <int _veclen>
struct Policy4x4<double, _veclen> {
  typedef KernelPolicy<double, _veclen, 16, 4, 4, 16, 16> Policy;
};

template <typename DataT, typename OutT, typename IdxT, bool Sqrt,
          typename Policy, typename ReduceOpT>
struct FusedL2NN {
 private:
  typedef Policy P;

  IdxT m, n, k, xrowid, yrowid;
  DataT *x, *y, *xn, *yn;
  OutT* min;
  int* mutex;

  int srowid, scolid;
  int accrowid, acccolid;

  DataT *sx, *sy;
  DataT *sxNorm, *syNorm;
  cub::KeyValuePair<IdxT, DataT>* sRed;
  int pageWr, pageRd;

  DataT maxVal;

  DataT acc[P::AccRowsPerTh][P::AccColsPerTh];
  DataT regx[P::AccRowsPerTh][P::Veclen], regy[P::AccColsPerTh][P::Veclen];
  DataT ldgDataX[P::LdgPerThX][P::Veclen], ldgDataY[P::LdgPerThY][P::Veclen];

  ReduceOpT redOp;

#if (ENABLE_MEMCPY_ASYNC == 1)
  nvcuda::experimental::pipeline pipe;
#endif

  static const DataT Zero = (DataT)0;
  static const DataT Two = (DataT)2.0;

 public:
  DI FusedL2NN(OutT* _min, DataT* _x, DataT* _y, DataT* _xn, DataT* _yn,
               IdxT _m, IdxT _n, IdxT _k, char* _smem, DataT _mv, int* _mut,
               ReduceOpT op)
    : m(_m),
      n(_n),
      k(_k),
      xrowid(IdxT(blockIdx.x) * P::Mblk + threadIdx.x / P::LdgThK),
      yrowid(IdxT(blockIdx.y) * P::Nblk + threadIdx.x / P::LdgThK),
      x(_x + xrowid * k),
      y(_y + yrowid * k),
      xn(_xn),
      yn(_yn),
      min(_min),
      mutex(_mut),
      srowid(threadIdx.x / P::LdgThK),
      scolid((threadIdx.x % P::LdgThK) * P::Veclen),
      accrowid(threadIdx.x / P::AccThCols),
      acccolid(threadIdx.x % P::AccThCols),
      sx((DataT*)_smem),
      sy(&(sx[P::SmemPageX])),
      sxNorm((DataT*)_smem),
      syNorm(&(sxNorm[P::Mblk])),
      sRed((cub::KeyValuePair<IdxT, DataT>*)_smem),
      pageWr(0),
      pageRd(0),
      maxVal(_mv),
      redOp(op) {}

  DI void run() {
    prolog();
    loop();
    __syncthreads();  // so that we can safely reuse smem
    epilog();
  }

 private:
  DI void prolog() {
    ldgXY(0);
#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < P::AccColsPerTh; ++j) {
        acc[i][j] = Zero;
      }
    }
    stsXY();
    __syncthreads();
    pageWr ^= 1;
  }

  DI void loop() {
    for (int kidx = P::Kblk; kidx < k; kidx += P::Kblk) {
      ldgXY(kidx);
      accumulate();
      stsXY();
      __syncthreads();
      pageWr ^= 1;
      pageRd ^= 1;
    }
    accumulate();  // last iteration
  }

  DI void epilog() {
    for (int i = threadIdx.x; i < P::Mblk; i += P::Nthreads) {
      auto idx = blockIdx.x * P::Mblk + i;
      sxNorm[i] = idx < m ? xn[idx] : maxVal;
    }
    for (int i = threadIdx.x; i < P::Nblk; i += P::Nthreads) {
      auto idx = blockIdx.y * P::Nblk + i;
      syNorm[i] = idx < n ? yn[idx] : maxVal;
    }
    __syncthreads();
    DataT regxn[P::AccRowsPerTh], regyn[P::AccColsPerTh];
#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
      regxn[i] = sxNorm[i * P::AccThRows + accrowid];
    }
#pragma unroll
    for (int i = 0; i < P::AccColsPerTh; ++i) {
      regyn[i] = syNorm[i * P::AccThCols + acccolid];
    }
#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < P::AccColsPerTh; ++j) {
        acc[i][j] = regxn[i] + regyn[j] - Two * acc[i][j];
      }
    }
    if (Sqrt) {
#pragma unroll
      for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
        for (int j = 0; j < P::AccColsPerTh; ++j) {
          acc[i][j] = mySqrt(acc[i][j]);
        }
      }
    }
    // reduce
    cub::KeyValuePair<IdxT, DataT> val[P::AccRowsPerTh];
    KVPMinReduce<IdxT, DataT> pairRedOp;
    auto lid = laneId();
#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
      val[i] = {-1, maxVal};
#pragma unroll
      for (int j = 0; j < P::AccColsPerTh; ++j) {
        auto tmpkey = acccolid + j * P::AccThCols + blockIdx.y * P::Nblk;
        cub::KeyValuePair<IdxT, DataT> tmp = {tmpkey, acc[i][j]};
        if (tmpkey < n) val[i] = pairRedOp(tmp, val[i]);
      }
      __syncthreads();
#pragma unroll
      for (int j = P::AccThCols / 2; j > 0; j >>= 1) {
        auto tmpkey = shfl(val[i].key, lid + j);
        auto tmpvalue = shfl(val[i].value, lid + j);
        cub::KeyValuePair<IdxT, DataT> tmp = {tmpkey, tmpvalue};
        val[i] = pairRedOp(tmp, val[i]);
      }
    }
    if (lid % P::AccThCols == 0) {
#pragma unroll
      for (int i = 0; i < P::AccRowsPerTh; ++i) {
        sRed[i * P::AccThCols + accrowid] = val[i];
      }
    }
    __syncthreads();
    updateResults();
  }

  DI void updateResults() {
    /**
     * @todo: From Volta onwards see if "coalesced" atomicCAS approach as
     *        written below helps improve perf
     * <code>
     *   auto tid = threadIdx.x;
     *   auto rid = IdxT(blockIdx.x) * P::Mblk + tid;
     *   if (rid < m) {
     *     auto val = sRed[i];
     *     while (atomicCAS(mutex + rid, 0, 1) == 1)
     *       ;
     *     __threadfence();
     *     redOp(min + rid, val);
     *     __threadfence();
     *     atomicCAS(mutex + rid, 1, 0);
     *   }
     * </code>
     */
    // for now have first lane from each warp update a unique output row. This
    // will resolve hang issues with pre-Volta architectures
    auto nWarps = blockDim.x / WarpSize;
    auto lid = laneId();
    auto ridx = IdxT(blockIdx.x) * P::Mblk;
    if (lid == 0) {
      for (int i = threadIdx.x / WarpSize; i < P::Mblk; i += nWarps) {
        auto rid = ridx + i;
        if (rid < m) {
          auto val = sRed[i];
          while (atomicCAS(mutex + rid, 0, 1) == 1)
            ;
          __threadfence();
          redOp(min + rid, val);
          __threadfence();
          atomicCAS(mutex + rid, 1, 0);
        }
      }
    }
  }

  DI void accumulate() {
#pragma unroll
    for (int ki = 0; ki < P::Kblk; ki += P::Veclen) {
      ldsXY(ki);
#pragma unroll
      for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
        for (int j = 0; j < P::AccColsPerTh; ++j) {
#pragma unroll
          for (int v = 0; v < P::Veclen; ++v) {
            acc[i][j] += regx[i][v] * regy[j][v];
          }
        }
      }
    }
  }

#if (ENABLE_MEMCPY_ASYNC == 1)
  ///@todo: fix this to use memcpy_async
  DI void ldgXY(IdxT kidx) {
    auto koffset = kidx + scolid;
    for (int i = 0; i < P::LdgPerThX; ++i) {
      if (koffset < k && (xrowid + i * P::LdgRowsX) < m) {
        ldg(ldgDataX[i], x + i * P::LdgRowsX * k + koffset);
      } else {
#pragma unroll
        for (int j = 0; j < P::Veclen; ++j) {
          ldgDataX[i][j] = Zero;
        }
      }
    }
    for (int i = 0; i < P::LdgPerThY; ++i) {
      if (koffset < k && (yrowid + i * P::LdgRowsY) < n) {
        ldg(ldgDataY[i], y + i * P::LdgRowsY * k + koffset);
      } else {
#pragma unroll
        for (int j = 0; j < P::Veclen; ++j) {
          ldgDataY[i][j] = Zero;
        }
      }
    }
  }
#else  // ENABLE_MEMCPY_ASYNC
  DI void ldgXY(IdxT kidx) {
    auto koffset = kidx + scolid;
    for (int i = 0; i < P::LdgPerThX; ++i) {
      if (koffset < k && (xrowid + i * P::LdgRowsX) < m) {
        ldg(ldgDataX[i], x + i * P::LdgRowsX * k + koffset);
      } else {
#pragma unroll
        for (int j = 0; j < P::Veclen; ++j) {
          ldgDataX[i][j] = Zero;
        }
      }
    }
    for (int i = 0; i < P::LdgPerThY; ++i) {
      if (koffset < k && (yrowid + i * P::LdgRowsY) < n) {
        ldg(ldgDataY[i], y + i * P::LdgRowsY * k + koffset);
      } else {
#pragma unroll
        for (int j = 0; j < P::Veclen; ++j) {
          ldgDataY[i][j] = Zero;
        }
      }
    }
  }
#endif  // ENABLE_MEMCPY_ASYNC

#if (ENABLE_MEMCPY_ASYNC == 1)
  ///@todo: fix this to be a no-op in case of memcpy_async
  DI void stsXY() {
    auto offset = pageWr * P::SmemPage + srowid * P::SmemStride + scolid;
    auto* saddrx = sx + offset;
#pragma unroll
    for (int i = 0; i < P::LdgPerThX; ++i) {
      sts(saddrx + i * P::LdgRowsX * P::SmemStride, ldgDataX[i]);
    }
    auto* saddry = sy + offset;
#pragma unroll
    for (int i = 0; i < P::LdgPerThY; ++i) {
      sts(saddry + i * P::LdgRowsY * P::SmemStride, ldgDataY[i]);
    }
  }
#else  // ENABLE_MEMCPY_ASYNC
  DI void stsXY() {
    auto offset = pageWr * P::SmemPage + srowid * P::SmemStride + scolid;
    auto* saddrx = sx + offset;
#pragma unroll
    for (int i = 0; i < P::LdgPerThX; ++i) {
      sts(saddrx + i * P::LdgRowsX * P::SmemStride, ldgDataX[i]);
    }
    auto* saddry = sy + offset;
#pragma unroll
    for (int i = 0; i < P::LdgPerThY; ++i) {
      sts(saddry + i * P::LdgRowsY * P::SmemStride, ldgDataY[i]);
    }
  }
#endif  // ENABLE_MEMCPY_ASYNC

  DI void ldsXY(int kidx) {
    ldsX(kidx, sx + pageRd * P::SmemPage);
    ldsY(kidx, sy + pageRd * P::SmemPage);
  }

  DI void ldsX(int kidx, DataT* smem) {
    auto* saddr = smem + accrowid * P::SmemStride + kidx;
#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
      lds(regx[i], saddr + i * P::AccThRows * P::SmemStride);
    }
  }

  DI void ldsY(int kidx, DataT* smem) {
    auto* saddr = smem + acccolid * P::SmemStride + kidx;
#pragma unroll
    for (int i = 0; i < P::AccColsPerTh; ++i) {
      lds(regy[i], saddr + i * P::AccThCols * P::SmemStride);
    }
  }
};  // struct FusedL2NN

template <typename DataT, typename OutT, typename IdxT, bool Sqrt,
          typename Policy, typename ReduceOpT>
__global__ __launch_bounds__(Policy::Nthreads, 2) void fusedL2NNkernel(
  OutT* min, DataT* x, DataT* y, DataT* xn, DataT* yn, IdxT m, IdxT n, IdxT k,
  DataT maxVal, int* mutex, ReduceOpT redOp) {
  extern __shared__ char smem[];
  FusedL2NN<DataT, OutT, IdxT, Sqrt, Policy, ReduceOpT> obj(
    min, x, y, xn, yn, m, n, k, smem, maxVal, mutex, redOp);
  obj.run();
}

template <typename DataT, typename OutT, typename IdxT, typename ReduceOpT>
__global__ void initKernel(OutT* min, IdxT m, DataT maxVal, ReduceOpT redOp) {
  auto tid = IdxT(blockIdx.x) * blockDim.x + threadIdx.x;
  if (tid < m) {
    redOp.init(min + tid, maxVal);
  }
}

template <typename DataT, typename OutT, typename IdxT, int VecLen,
          typename ReduceOpT>
void fusedL2NNImpl(OutT* min, DataT* x, DataT* y, DataT* xn, DataT* yn, IdxT m,
                   IdxT n, IdxT k, int* workspace, ReduceOpT redOp, bool sqrt,
                   bool initOutBuffer, cudaStream_t stream) {
  typedef typename Policy4x4<DataT, VecLen>::Policy Policy;
  dim3 grid(ceildiv<int>(m, Policy::Mblk), ceildiv<int>(n, Policy::Nblk));
  dim3 blk(Policy::Nthreads);
  auto nblks = ceildiv<int>(m, Policy::Nthreads);
  auto maxVal = std::numeric_limits<DataT>::max();
  CUDA_CHECK(cudaMemsetAsync(workspace, 0, sizeof(int) * m, stream));
  if (initOutBuffer) {
    initKernel<DataT, OutT, IdxT, ReduceOpT>
      <<<nblks, Policy::Nthreads, 0, stream>>>(min, m, maxVal, redOp);
    CUDA_CHECK(cudaGetLastError());
  }
  if (sqrt) {
    fusedL2NNkernel<DataT, OutT, IdxT, true, Policy, ReduceOpT>
      <<<grid, blk, Policy::SmemSize, stream>>>(min, x, y, xn, yn, m, n, k,
                                                maxVal, workspace, redOp);
  } else {
    fusedL2NNkernel<DataT, OutT, IdxT, false, Policy, ReduceOpT>
      <<<grid, blk, Policy::SmemSize, stream>>>(min, x, y, xn, yn, m, n, k,
                                                maxVal, workspace, redOp);
  }
  CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Fused L2 distance and 1-nearest-neighbor computation in a single call.
 *        The benefits of such a call are 2-fold: 1) eliminate the need for an
 *        intermediate buffer to store the output of gemm 2) reduce the memory
 *        read traffic on this intermediate buffer, otherwise needed during the
 *        reduction phase for 1-NN.
 * @tparam DataT data type
 * @tparam OutT output type to either store 1-NN indices and their min distances
 *              or store only the min distances. Accordingly, one has to pass an
 *              appropriate `ReduceOpT`
 * @tparam IdxT indexing arithmetic type
 * @tparam ReduceOpT A struct to perform the final needed reduction operation
 *                   and also to initialize the output array elements with the
 *                   appropriate initial value needed for reduction.
 * @param[out] min will contain the reduced output (Length = `m`) (on device)
 * @param[in] x first matrix. Row major. Dim = `m x k`. (on device).
 * @param[in] y second matrix. Row major. Dim = `n x k`. (on device).
 * @param[in] xn L2 squared norm of `x`. Length = `m`. (on device).
 * @param[in] yn L2 squared norm of `y`. Length = `n`. (on device).
 * @param[in] m gemm m
 * @param[in] n gemm n
 * @param[in] k gemm k
 * @param[in] workspace temporary workspace. Size = sizeof(int)*m. (on device)
 * @param[in] sqrt Whether the output `minDist` should contain L2-sqrt or not
 * @param[in] initOutBuffer whether to initialize the output buffer before the
 *                          main kernel launch
 * @param[in] stream cuda stream
 */
template <typename DataT, typename OutT, typename IdxT, typename ReduceOpT>
void fusedL2NN(OutT* min, DataT* x, DataT* y, DataT* xn, DataT* yn, IdxT m,
               IdxT n, IdxT k, void* workspace, ReduceOpT redOp, bool sqrt,
               bool initOutBuffer, cudaStream_t stream) {
  size_t bytes = sizeof(DataT) * k;
  if (16 % sizeof(DataT) == 0 && bytes % 16 == 0) {
    fusedL2NNImpl<DataT, OutT, IdxT, 16 / sizeof(DataT), ReduceOpT>(
      min, x, y, xn, yn, m, n, k, (int*)workspace, redOp, sqrt, initOutBuffer,
      stream);
  } else if (8 % sizeof(DataT) == 0 && bytes % 8 == 0) {
    fusedL2NNImpl<DataT, OutT, IdxT, 8 / sizeof(DataT), ReduceOpT>(
      min, x, y, xn, yn, m, n, k, (int*)workspace, redOp, sqrt, initOutBuffer,
      stream);
  } else {
    fusedL2NNImpl<DataT, OutT, IdxT, 1, ReduceOpT>(min, x, y, xn, yn, m, n, k,
                                                   (int*)workspace, redOp, sqrt,
                                                   initOutBuffer, stream);
  }
}

}  // namespace Distance
}  // namespace MLCommon

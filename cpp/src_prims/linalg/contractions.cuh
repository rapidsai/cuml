/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

namespace MLCommon {

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

namespace LinAlg {

/**
 * @brief This is the central enum that should be used to configure the perf
 *        landscape of the Contraction kernel.
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
 *             This also determines the number of threads per block
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

/**
 * @brief Base class for gemm-like NT contractions
 */
template <typename DataT, typename IdxT, typename Policy>
struct Contractions_NT {
 protected:
  typedef Policy P;

  IdxT m, n, k, xrowid, yrowid;
  DataT *x, *y;

  int srowid, scolid;
  int accrowid, acccolid;

  DataT *sx, *sy;
  int pageWr, pageRd;

  static const DataT Zero = (DataT)0;

 public:
  DI Contractions_NT(DataT* _x, DataT* _y, IdxT _m, IdxT _n, IdxT _k,
                     char* _smem)
    : m(_m),
      n(_n),
      k(_k),
      xrowid(IdxT(blockIdx.x) * P::Mblk + threadIdx.x / P::LdgThK),
      yrowid(IdxT(blockIdx.y) * P::Nblk + threadIdx.x / P::LdgThK),
      x(_x + xrowid * k),
      y(_y + yrowid * k),
      srowid(threadIdx.x / P::LdgThK),
      scolid((threadIdx.x % P::LdgThK) * P::Veclen),
      accrowid(threadIdx.x / P::AccThCols),
      acccolid(threadIdx.x % P::AccThCols),
      sx((DataT*)_smem),
      sy(&(sx[P::SmemPageX])),
      pageWr(0),
      pageRd(0) {}

 protected:
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

  DI void ldsXY(int kidx) {
    ldsX(kidx, sx + pageRd * P::SmemPage);
    ldsY(kidx, sy + pageRd * P::SmemPage);
  }

 private:
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
};  // struct Contractions_NT

}  // namespace LinAlg
}  // namespace MLCommon

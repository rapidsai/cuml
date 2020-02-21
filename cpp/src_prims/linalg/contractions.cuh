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

#include <common/device_ld_st.cuh>

namespace MLCommon {
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

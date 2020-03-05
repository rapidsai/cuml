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

#include <common/device_loads_stores.cuh>

namespace MLCommon {
namespace LinAlg {

/**
 * @brief This is the central enum that should be used to configure the perf
 *        landscape of the Contraction kernel.
 *
 * Main goal of this Policy struct is to provide sufficient knobs to tune the
 * perf of Contraction kernel, as and when we see matrices of different shapes.
 *
 * @tparam DataT   the IO and math datatype
 * @tparam _veclen number of k-elements loaded by each thread for every LDG call
 *                 it makes. This should be configured based on the input 'k'
 *                 value and the input data type. For eg: if DataT = float and
 *                 k is multiples of 4, then setting this to 4 gives the best
 *                 LDG pattern. Possible values are {1, 2, 4}.
 * @tparam _kblk   number of k-elements operated upon per main-loop iteration.
 *                 Therefore total number of main-loop iterations will be
 *                 `ceil(k/_kblk)`. This must be multiples of `_veclen`. Do note
 *                 that bigger this value, the greater shared mem requirement.
 * @tparam _rpt    Defines the number of rows that a given thread accumulates on.
 *                 This directly results in increased register pressure. This
 *                 also is used to compute the number of m-elements worked upon
 *                 by each thread block.
 * @tparam _rpt    Defines the number of cols that a given thread accumulates on.
 *                 This directly results in increased register pressure. This
 *                 also is used to compute the number of n-elements worked upon
 *                 by each thread block.
 * @tparam _tr     Number of threads working on the same output column. This is
 *                 used to compute the number of m-elements worked upon by each
 *                 thread block. This also determines the number of threads per
 *                 thread block
 * @tparam _tc     Number of threads working on the same output row. This is
 *                 used to compute the number of m-elements worked upon by each
 *                 thread block. This also determines the number of threads per
 *                 thread block
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

/**
 * @defgroup Policy4x4 16 elements per thread Policy with k-block = 32
 * @{
 */
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
/** @} */

/**
 * @brief Base class for gemm-like NT contractions
 *
 * This class does not provide any arithmetic operations, but only provides the
 * memory-related operations of loading the `x` and `y` matrix blocks from the
 * global memory into shared memory and then from shared into registers. Thus,
 * this class acts as a basic building block for further composing gemm-like NT
 * contractions on input matrices which are row-major (and so does the output)
 *
 * @tparam DataT  IO and math data type
 * @tparam IdxT   indexing type
 * @tparam Policy policy used to customize memory access behavior.
 *                See documentation for `KernelPolicy` to know more.
 */
template <typename DataT, typename IdxT, typename Policy>
struct Contractions_NT {
 protected:
  typedef Policy P;

  /** number of rows in X */
  IdxT m;
  /** number of rows in Y */
  IdxT n;
  /** number of columns in X and Y */
  IdxT k;
  /** current thread's global mem row id for X data */
  IdxT xrowid;
  /** current thread's global mem row id for Y data */
  IdxT yrowid;
  /** global memory pointer to X matrix */
  const DataT* x;
  /** global memory pointer to Y matrix */
  const DataT* y;

  /** current thread's smem row id */
  int srowid;
  /** current thread's smem column id */
  int scolid;
  /** current thread's accumulation row id */
  int accrowid;
  /** current thread's accumulation column id */
  int acccolid;

  /** base smem pointer for X data storage */
  DataT* sx;
  /** base smem pointer for Y data storage */
  DataT* sy;
  /** index pointing the correct smem page for writing after `ldgXY()` */
  int pageWr;
  /** index pointing the correct smem page for reading during `ldsXY()` */
  int pageRd;

  /** block of X data loaded from smem after `ldsXY()` */
  DataT regx[P::AccRowsPerTh][P::Veclen];
  /** block of Y data loaded from smem after `ldsXY()` */
  DataT regy[P::AccColsPerTh][P::Veclen];
  /** block of X data loaded from global mem after `ldgXY()` */
  DataT ldgDataX[P::LdgPerThX][P::Veclen];
  /** block of Y data loaded from global mem after `ldgXY()` */
  DataT ldgDataY[P::LdgPerThY][P::Veclen];

  static const DataT Zero = (DataT)0;

 public:
  /**
   * @brief Ctor
   * @param[in] _x X matrix. [on device] [dim = _m x _k] [row-major]
   * @param[in] _y Y matrix. [on device] [dim = _n x _k] [row-major]
   * @param[in] _m number of rows of X
   * @param[in] _n number of rows of Y
   * @param[in] _k number of cols of X and Y
   * @param[in] _smem shared memory region used during computations
   */
  DI Contractions_NT(const DataT* _x, const DataT* _y, IdxT _m, IdxT _n,
                     IdxT _k, char* _smem)
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
  /**
   * @brief Load current block of X/Y from global memory to smem
   * @param[in] kidx current start index of k to be loaded
   */
  DI void ldgsts(IdxT kidx) {
    ldgstsX(kidx, sx + pageWr * P::SmemPage);
    ldgstsY(kidx, sy + pageWr * P::SmemPage);
  }

  /**
   * @brief Load X and Y block from shared memory to registers
   * @param[in] kidx k value from the current k-block to be loaded from smem
   */
  DI void ldsXY(int kidx) {
    ldsX(kidx, sx + pageRd * P::SmemPage);
    ldsY(kidx, sy + pageRd * P::SmemPage);
  }

 private:
  DI void ldgstsX(IdxT kidx, DataT* smem) {
    DataT data[P::LdgPerThX][P::Veclen];
    // LDG
    auto koffset = kidx + scolid;
    for (int i = 0; i < P::LdgPerThX; ++i) {
      if (koffset < k && (xrowid + i * P::LdgRowsX) < m) {
        ldg(data[i], x + i * P::LdgRowsX * k + koffset);
      } else {
#pragma unroll
        for (int j = 0; j < P::Veclen; ++j) {
          data[i][j] = Zero;
        }
      }
    }
    // STS
    auto* saddr = smem + srowid * P::SmemStride + scolid;
#pragma unroll
    for (int i = 0; i < P::LdgPerThX; ++i) {
      sts(saddr + i * P::LdgRowsX * P::SmemStride, data[i]);
    }
  }

  DI void ldgstsY(IdxT kidx, DataT* smem) {
    DataT data[P::LdgPerThX][P::Veclen];
    // LDG
    auto koffset = kidx + scolid;
    for (int i = 0; i < P::LdgPerThY; ++i) {
      if (koffset < k && (yrowid + i * P::LdgRowsY) < n) {
        ldg(data[i], y + i * P::LdgRowsY * k + koffset);
      } else {
#pragma unroll
        for (int j = 0; j < P::Veclen; ++j) {
          data[i][j] = Zero;
        }
      }
    }
    // STS
    auto* saddr = smem + srowid * P::SmemStride + scolid;
#pragma unroll
    for (int i = 0; i < P::LdgPerThY; ++i) {
      sts(saddr + i * P::LdgRowsY * P::SmemStride, data[i]);
    }
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

  ///@todo: splitting ldg/sts phases can give more latency hiding opportunity
  ///       thereby improving perf. However the below code is causing kmeans
  ///       unit-tests to fail. Will need to fix this issue to enable the below
  ///       code-blocks here and also in FusedL2NN class
  //   DI void ldgXY(IdxT kidx) {
  //     auto koffset = kidx + scolid;
  //     for (int i = 0; i < P::LdgPerThX; ++i) {
  //       if (koffset < k && (xrowid + i * P::LdgRowsX) < m) {
  //         ldg(ldgDataX[i], x + i * P::LdgRowsX * k + koffset);
  //       } else {
  // #pragma unroll
  //         for (int j = 0; j < P::Veclen; ++j) {
  //           ldgDataX[i][j] = Zero;
  //         }
  //       }
  //     }
  //     for (int i = 0; i < P::LdgPerThY; ++i) {
  //       if (koffset < k && (yrowid + i * P::LdgRowsY) < n) {
  //         ldg(ldgDataY[i], y + i * P::LdgRowsY * k + koffset);
  //       } else {
  // #pragma unroll
  //         for (int j = 0; j < P::Veclen; ++j) {
  //           ldgDataY[i][j] = Zero;
  //         }
  //       }
  //     }
  //   }

  //   DI void stsXY() {
  //     auto offset = pageWr * P::SmemPage + srowid * P::SmemStride + scolid;
  //     auto* saddrx = sx + offset;
  // #pragma unroll
  //     for (int i = 0; i < P::LdgPerThX; ++i) {
  //       sts(saddrx + i * P::LdgRowsX * P::SmemStride, ldgDataX[i]);
  //     }
  //     auto* saddry = sy + offset;
  // #pragma unroll
  //     for (int i = 0; i < P::LdgPerThY; ++i) {
  //       sts(saddry + i * P::LdgRowsY * P::SmemStride, ldgDataY[i]);
  //     }
  //   }
};  // struct Contractions_NT

}  // namespace LinAlg
}  // namespace MLCommon

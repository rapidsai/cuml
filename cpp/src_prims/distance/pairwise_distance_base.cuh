/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <raft/linalg/contractions.cuh>
#include <raft/linalg/norm.cuh>

namespace MLCommon {
namespace Distance {

/**
 * @brief Device class for L1, L2 and cosine distance metrics.
 * @tparam useNorms       whether norms are needed
 * @tparam DataT          input data-type (for A and B matrices)
 * @tparam AccT           accumulation data-type
 * @tparam OutT           output data-type (for C and D matrices)
 * @tparam IdxT           index data-type
 * @tparam Policy         struct which tunes the Contraction kernel
 * @tparam CoreLambda     tells how to accumulate an x and y into 
                          acc. its signature:
    template <typename AccT, typename DataT> void core_lambda(AccT& acc,
      const DataT& x, const DataT& y)
 * @tparam EpilogueLambda applies an elementwise function to compute final 
    values. Its signature is:
    template <typename AccT, typename DataT> void epilogue_lambda
    (AccT acc[][], DataT* regxn, DataT* regyn);
 * @tparam FinalLambda the final lambda called on final distance value
 * @param[in] x input matrix
 * @param[in] y input matrix
 * @param[in] m number of rows of A and C/D
 * @param[in] n number of columns of B and C/D
 * @param[in] k number of cols of A and rows of B
 * @param[in] xn row norms of input matrix A. Required for expanded L2, cosine
 * @param[in] yn row norms of input matrix B. Required for expanded L2, cosine
 * @param[output] pD output matrix
 * @param[in] smem shared mem buffer for intermediate storage of A, B, xn & yn.
 * @param core_op the core accumulation operation lambda
 * @param epilog_op the epilog operation lambda
 * @param fin_op the final gemm epilogue lambda
 */
template <bool useNorms, typename DataT, typename AccT, typename OutT,
          typename IdxT, typename Policy, typename CoreLambda,
          typename EpilogueLambda, typename FinalLambda,
          typename BaseClass =
            raft::linalg::Contractions_NT<DataT, IdxT, Policy>>
struct PairwiseDistances : public BaseClass {
 private:
  typedef Policy P;
  const DataT* xn;
  const DataT* yn;
  DataT* sxNorm;
  DataT* syNorm;
  OutT* dOutput;
  char* smem;
  CoreLambda core_op;
  EpilogueLambda epilog_op;
  FinalLambda fin_op;

  AccT acc[P::AccRowsPerTh][P::AccColsPerTh];

 public:
  // Constructor
  DI PairwiseDistances(const DataT* _x, const DataT* _y, IdxT _m, IdxT _n,
                       IdxT _k, const DataT* _xn, const DataT* _yn,
                       OutT* _dOutput, char* _smem, CoreLambda _core_op,
                       EpilogueLambda _epilog_op, FinalLambda _fin_op)
    : BaseClass(_x, _y, _m, _n, _k, _smem),
      sxNorm((DataT*)_smem),
      syNorm(&(sxNorm[P::Mblk])),
      xn(_xn),
      yn(_yn),
      dOutput(_dOutput),
      smem(_smem),
      core_op(_core_op),
      epilog_op(_epilog_op),
      fin_op(_fin_op) {}

  DI void run() {
    prolog();
    loop();
    epilog();
  }

 private:
  DI void prolog() {
    this->ldgXY(0);
#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < P::AccColsPerTh; ++j) {
        acc[i][j] = BaseClass::Zero;
      }
    }
    this->stsXY();
    __syncthreads();
    this->pageWr ^= 1;
  }

  DI void loop() {
    for (int kidx = P::Kblk; kidx < this->k; kidx += P::Kblk) {
      this->ldgXY(kidx);
      accumulate();  // on the previous k-block
      this->stsXY();
      __syncthreads();
      this->pageWr ^= 1;
      this->pageRd ^= 1;
    }
    accumulate();  // last iteration
  }

  DI void accumulate() {
#pragma unroll
    for (int ki = 0; ki < P::Kblk; ki += P::Veclen) {
      this->ldsXY(ki);
#pragma unroll
      for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
        for (int j = 0; j < P::AccColsPerTh; ++j) {
#pragma unroll
          for (int v = 0; v < P::Veclen; ++v) {
            core_op(acc[i][j], this->regx[i][v], this->regy[j][v]);
          }
        }
      }
    }
  }

  DI void epilog() {
    if (useNorms) {
      __syncthreads();  // so that we can safely reuse smem

      // Load x & y norms required by this threadblock in shmem buffer
      for (int i = threadIdx.x; i < P::Mblk; i += P::Nthreads) {
        auto idx = blockIdx.x * P::Mblk + i;
        sxNorm[i] = idx < this->m ? xn[idx] : 0;
      }
      for (int i = threadIdx.x; i < P::Nblk; i += P::Nthreads) {
        auto idx = blockIdx.y * P::Nblk + i;
        syNorm[i] = idx < this->n ? yn[idx] : 0;
      }
      __syncthreads();

      DataT regxn[P::AccRowsPerTh], regyn[P::AccColsPerTh];
#pragma unroll
      for (int i = 0; i < P::AccRowsPerTh; ++i) {
        regxn[i] = sxNorm[i * P::AccThRows + (threadIdx.x / P::AccThCols)];
      }
#pragma unroll
      for (int i = 0; i < P::AccColsPerTh; ++i) {
        regyn[i] = syNorm[i * P::AccThCols + (threadIdx.x % P::AccThCols)];
      }

      epilog_op(acc, regxn, regyn);
    } else {
      epilog_op(acc, nullptr, nullptr);
    }

    IdxT startx = blockIdx.x * P::Mblk + this->accrowid;
    IdxT starty = blockIdx.y * P::Nblk + this->acccolid;
#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
      auto rowId = startx + i * P::AccThRows;
#pragma unroll
      for (int j = 0; j < P::AccColsPerTh; ++j) {
        auto colId = starty + j * P::AccThCols;
        if (rowId < this->m && colId < this->n) {
          dOutput[rowId * this->n + colId] = fin_op(acc[i][j], 0);
        }
      }
    }
  }
};  // struct PairwiseDistances

/**
 * @brief the distance matrix calculation kernel for L1, L2 and cosine
 * @tparam useNorms       whether norms are needed
 * @tparam DataT          input data-type (for A and B matrices)
 * @tparam AccT           accumulation data-type
 * @tparam OutT           output data-type (for C and D matrices)
 * @tparam IdxT           index data-type
 * @tparam Policy         struct which tunes the Contraction kernel
 * @tparam CoreLambda     lambda which implements accumulation operation
 * @tparam EpilogueLambda lambda which implements operation for calculating
                          final value.
 * @tparam FinalLambda    final lambda called on final distance value
 *
 * @param[in]       x input matrix
 * @param[in]       y input matrix
 * @param[in]       xn row norms of input matrix A.
 * @param[in]       yn row norms of input matrix B.
 * @param[in]       m number of rows of A and C/D
 * @param[in]       n number of columns of B and C/D
 * @param[in]       k number of cols of A and rows of B
 * @param[output]   pD output matrix
 * @param core_op   the core lambda
 * @param epilog_op the epilogue lambda
 * @param fin_op    the final gemm epilogue lambda
 */
template <bool useNorms, typename DataT, typename AccT, typename OutT,
          typename IdxT, typename Policy, typename CoreLambda,
          typename EpilogueLambda, typename FinalLambda>
__global__ __launch_bounds__(
  Policy::Nthreads,
  2) void pairwiseDistanceMatKernel(const DataT* x, const DataT* y,
                                    const DataT* _xn, const DataT* _yn, IdxT m,
                                    IdxT n, IdxT k, OutT* dOutput,
                                    CoreLambda core_op,
                                    EpilogueLambda epilog_op,
                                    FinalLambda fin_op) {
  extern __shared__ char smem[];

  PairwiseDistances<useNorms, DataT, AccT, OutT, IdxT, Policy, CoreLambda,
                    EpilogueLambda, FinalLambda>
    obj(x, y, m, n, k, _xn, _yn, dOutput, smem, core_op, epilog_op, fin_op);
  obj.run();
}

};  // end namespace Distance
};  // end namespace MLCommon
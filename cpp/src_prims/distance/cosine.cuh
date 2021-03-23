/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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
#include <linalg/eltwise2d.cuh>
#include "distance_fragment_multiply_add.cuh"
#include "pairwise_distance_base.cuh"

#include <linalg/cutlass_gemm.cuh>
#include <raft/linalg/norm.cuh>
#include "distance_epilogue.cuh"
#include "distance_epilogue_functor.cuh"
#include "distance_epilogue_traits.h"

#include <cutlass/gemm/gemm_epilogue_traits.h>
#include <cutlass/gemm/thread_multiply_add.h>
#include <cutlass/shape.h>

#include <type_traits>

namespace MLCommon {
namespace Distance {

/**
 * @brief the cosine distance matrix calculation implementer
 *  It computes the following equation: 
 *    C = 1 - op(A * B / sqrt(A^2) * sqrt(B^2)))
 * @tparam DataT input data-type (for A and B matrices)
 * @tparam AccT   accumulation data-type
 * @tparam OutT   output data-type (for C and D matrices)
 * @tparam IdxT   index data-type
 * @tparam Veclen number of k-elements loaded by each thread for every LDG call
 *                it makes. check contractions.cuh for details.
 * @tparam FinalLambda the final lambda called on final distance value
 * @param[in]     x input matrix
 * @param[in]     y input matrix
 * @param[in]     xn row norms of input matrix A.
 * @param[in]     yn row norms of input matrix B.
 * @param[in]     m number of rows of A and C/D
 * @param[in]     n number of columns of B and C/D
 * @param[in]     k number of cols of A and rows of B
 * @param[output] pD output matrix
 * @param fin_op  the final gemm epilogue lambda
*  @param stream  cuda stream to launch cuda operations.
 */
template <typename DataT, typename AccT, typename OutT, typename IdxT,
          int VecLen, typename FinalLambda>
void cosineImpl(const DataT *x, const DataT *y, const DataT *xn,
                const DataT *yn, IdxT m, IdxT n, IdxT k, OutT *dOutput,
                FinalLambda fin_op, cudaStream_t stream) {
  typedef typename raft::linalg::Policy4x4<DataT, VecLen>::Policy Policy;
  dim3 grid(raft::ceildiv<int>(m, Policy::Mblk),
            raft::ceildiv<int>(n, Policy::Nblk));
  dim3 blk(Policy::Nthreads);

  // Accumulation operation lambda
  auto core_lambda = [] __device__(AccT & acc, DataT & x, DataT & y) {
    acc += x * y;
  };

  // epilogue operation lambda for final value calculation
  auto epilog_lambda = [] __device__(
                         AccT acc[Policy::AccRowsPerTh][Policy::AccColsPerTh],
                         DataT * regxn, DataT * regyn) {
#pragma unroll
    for (int i = 0; i < Policy::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < Policy::AccColsPerTh; ++j) {
        acc[i][j] = acc[i][j] / (regxn[i] * regyn[j]);
      }
    }
  };

  pairwiseDistanceMatKernel<true, DataT, AccT, OutT, IdxT, Policy,
                            decltype(core_lambda), decltype(epilog_lambda),
                            FinalLambda>
    <<<grid, blk, Policy::SmemSize, stream>>>(
      x, y, xn, yn, m, n, k, dOutput, core_lambda, epilog_lambda, fin_op);

  CUDA_CHECK(cudaGetLastError());
}

template <typename DataT, typename AccT, typename OutT, typename IdxT,
          typename FinalLambda>
void cosine(IdxT m, IdxT n, IdxT k, const DataT *x, const DataT *y,
            const DataT *xn, const DataT *yn, OutT *dOutput, FinalLambda fin_op,
            cudaStream_t stream) {
  size_t bytes = sizeof(DataT) * k;
  if (16 % sizeof(DataT) == 0 && bytes % 16 == 0) {
    cosineImpl<DataT, AccT, OutT, IdxT, 16 / sizeof(DataT), FinalLambda>(
      x, y, xn, yn, m, n, k, dOutput, fin_op, stream);
  } else if (8 % sizeof(DataT) == 0 && bytes % 8 == 0) {
    cosineImpl<DataT, AccT, OutT, IdxT, 8 / sizeof(DataT), FinalLambda>(
      x, y, xn, yn, m, n, k, dOutput, fin_op, stream);
  } else {
    cosineImpl<DataT, AccT, OutT, IdxT, 1, FinalLambda>(
      x, y, xn, yn, m, n, k, dOutput, fin_op, stream);
  }
}

/**
 * @brief the expanded cosine distance matrix calculation
 *  It computes the following equation: 
 *              C = 1 - op(A * B / sqrt(A^2) * sqrt(B^2)))
 * @tparam IType input data-type (for A and B matrices)
 * @tparam AccType accumulation data-type
 * @tparam OType output data-type (for C and D matrices)
 * @tparam OutputTile_ output tile size for the thread block
 * @tparam FinalLambda user-defined epilogue lamba
 * @tparam Index_ Index type
 * @param m number of rows of A and C/D
 * @param n number of columns of B and C/D
 * @param k number of cols of A and rows of B
 * @param pA input matrix
 * @param pB input matrix
 * @param pD output matrix
 * @tparam in_params user-defined input parameter
 * @param workspace temporary workspace needed for computations
 * @param worksize number of bytes of the workspace
 * @param fin_op the final gemm epilogue lambda
 * @param stream cuda stream where to launch work
 * @param isRowMajor whether the input and output matrices are row major
 */
template <typename InType, typename AccType, typename OutType,
          typename OutputTile_, typename FinalLambda, typename Index_ = int>
void cosineAlgo1(Index_ m, Index_ n, Index_ k, const InType *pA,
                 const InType *pB, OutType *pD, AccType *workspace,
                 size_t worksize, FinalLambda fin_op, cudaStream_t stream,
                 bool isRowMajor) {
  auto norm_op = [] __device__(AccType in) { return raft::mySqrt(in); };

  // Wrap fin_op to allow computing 1 - pA before calling fin_op
  auto wrapped_fin_op = [fin_op] __device__(AccType d_val, Index_ g_d_idx) {
    return fin_op(static_cast<AccType>(1.0) - d_val, g_d_idx);
  };

  typedef std::is_same<OutType, bool> is_bool;

  ASSERT(!(((pA != pB) && (worksize < (m + n) * sizeof(AccType))) ||
           (worksize < m * sizeof(AccType))),
         "workspace size error");
  ASSERT(workspace != nullptr, "workspace is null");

  InType *col_vec = workspace;
  InType *row_vec = workspace;
  if (pA != pB) {
    row_vec += m;
    raft::linalg::rowNorm(col_vec, pA, k, m, raft::linalg::L2Norm, isRowMajor,
                          stream, norm_op);
    raft::linalg::rowNorm(row_vec, pB, k, n, raft::linalg::L2Norm, isRowMajor,
                          stream, norm_op);
  } else {
    raft::linalg::rowNorm(col_vec, pA, k, m, raft::linalg::L2Norm, isRowMajor,
                          stream, norm_op);
  }

  if (isRowMajor) {
    typedef typename std::conditional<is_bool::value, OutType, AccType>::type
      CosOutType;

    cosine<InType, AccType, CosOutType, Index_, decltype(wrapped_fin_op)>(
      m, n, k, pA, pB, col_vec, row_vec, reinterpret_cast<CosOutType *>(pD),
      wrapped_fin_op, stream);
  } else {
    typedef ExpandedDistanceFragmentMultiplyAdd<CosFusedDistance>
      FragmentMultiplyAdd_;
    typedef typename std::conditional<is_bool::value, AccType, OutType>::type
      EffOutType;
    EffOutType *pDCast =
      reinterpret_cast<EffOutType *>(pD);  // Pretend to be EffOutType;
    typedef typename cutlass::Shape<8, 8, 8> AccumulatorsPerThread_;
    typedef cutlass::gemm::ThreadMultiplyAdd<
      AccumulatorsPerThread_, cutlass::Shape<1, 4, 8>, InType, InType, AccType>
      MainLoopFunctor_;
    typedef LinAlg::CustomGemmConfig<InType, AccType, EffOutType, OutputTile_,
                                     AccumulatorsPerThread_, MainLoopFunctor_>
      GemmConfig_;

    typedef ExpandedDistanceEpilogueFunctor<InType, AccType, GemmConfig_,
                                            FragmentMultiplyAdd_>
      EpilogueFunctor_;

    typedef typename std::conditional<
      is_bool::value,
      BoolEpilogueTraitsHelper<GemmConfig_, EpilogueFunctor_, Index_>,
      cutlass::gemm::GemmEpilogueTraitsHelper<
        GemmConfig_, EpilogueFunctor_, Index_>>::type EpilogueTraitsHelper_;

    typedef typename cutlass::gemm::SimplifiedGemmEpilogueTraits<
      GemmConfig_, EpilogueFunctor_, Index_, EpilogueTraitsHelper_>
      GemmEpilogueTraits_;
    typedef ExpandedDistanceGemmEpilogue<GemmEpilogueTraits_> GemmEpilogue_;
    typedef typename EpilogueFunctor_::Params EpiParams;

    cublasOperation_t transa, transb;
    const InType *aPtr, *bPtr;
    Index_ lda, ldb, ldd;
    Index_ gemm_m, gemm_n;
    InType *rvec, *cvec;

    transa = CUBLAS_OP_N;
    transb = CUBLAS_OP_T;
    aPtr = pA;
    bPtr = pB;
    lda = m;
    ldb = n;
    ldd = m;
    gemm_m = m;
    gemm_n = n;
    cvec = row_vec;
    rvec = col_vec;

    LinAlg::gemm<InType, AccType, EffOutType, OutputTile_,
                 AccumulatorsPerThread_, MainLoopFunctor_, Index_, GemmConfig_,
                 EpilogueFunctor_, GemmEpilogueTraits_, GemmEpilogue_>(
      transa, transb, gemm_m, gemm_n, k, (EffOutType)1, aPtr, lda, bPtr, ldb,
      (EffOutType)0, nullptr, ldd, pDCast,
      [cvec, rvec] HD(EpiParams & p) {
        int err = p.initializeExtra(cvec, rvec, false);
        return err;
      },
      wrapped_fin_op, stream);
  }
}

};  // end namespace Distance
};  // end namespace MLCommon

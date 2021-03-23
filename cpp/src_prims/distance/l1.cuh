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
#include <linalg/custom_accum.h>
#include <linalg/cutlass_gemm.cuh>
#include <type_traits>

#include <raft/cuda_utils.cuh>
#include "pairwise_distance_base.cuh"

namespace MLCommon {
namespace Distance {

/**
 * @brief the L1 distance matrix calculation implementer
 *  It computes the following equation: cij = op(ai-bj)
 * @tparam DataT          input data-type (for A and B matrices)
 * @tparam AccT           accumulation data-type
 * @tparam OutT           output data-type (for C and D matrices)
 * @tparam IdxT           index data-type

 * @tparam FinalLambda    final lambda called on final distance value
 *
 * @param[in]       x input matrix
 * @param[in]       y input matrix
 * @param[in]       m number of rows of A and C/D
 * @param[in]       n number of columns of B and C/D
 * @param[in]       k number of cols of A and rows of B
 * @param[output]   pD output matrix
 * @param fin_op    the final gemm epilogue lambda
 */
template <typename DataT, typename AccT, typename OutT, typename IdxT,
          int VecLen, typename FinalLambda>
static void l1Impl(const DataT *x, const DataT *y, IdxT m, IdxT n, IdxT k,
                   OutT *dOutput, FinalLambda fin_op, cudaStream_t stream) {
  typedef typename raft::linalg::Policy4x4<DataT, VecLen>::Policy Policy;
  dim3 grid(raft::ceildiv<int>(m, Policy::Mblk),
            raft::ceildiv<int>(n, Policy::Nblk));
  dim3 blk(Policy::Nthreads);

  // Accumulation operation lambda
  auto core_lambda = [] __device__(AccT & acc, DataT & x, DataT & y) {
    const auto diff = raft::L1Op<AccT, IdxT>()(x - y);
    acc += diff;
  };

  // epilogue operation lambda for final value calculation
  auto epilog_lambda = [] __device__(
                         AccT acc[Policy::AccRowsPerTh][Policy::AccColsPerTh],
                         DataT * regxn, DataT * regyn) { return; };

  pairwiseDistanceMatKernel<false, DataT, AccT, OutT, IdxT, Policy,
                            decltype(core_lambda), decltype(epilog_lambda),
                            FinalLambda>
    <<<grid, blk, Policy::SmemSize, stream>>>(x, y, nullptr, nullptr, m, n, k,
                                              dOutput, core_lambda,
                                              epilog_lambda, fin_op);

  CUDA_CHECK(cudaGetLastError());
}

template <typename DataT, typename AccT, typename OutT, typename IdxT,
          typename FinalLambda>
void l1(IdxT m, IdxT n, IdxT k, const DataT *x, const DataT *y, OutT *dOutput,
        FinalLambda fin_op, cudaStream_t stream) {
  size_t bytes = sizeof(DataT) * k;
  if (16 % sizeof(DataT) == 0 && bytes % 16 == 0) {
    l1Impl<DataT, AccT, OutT, IdxT, 16 / sizeof(DataT), FinalLambda>(
      x, y, m, n, k, dOutput, fin_op, stream);
  } else if (8 % sizeof(DataT) == 0 && bytes % 8 == 0) {
    l1Impl<DataT, AccT, OutT, IdxT, 8 / sizeof(DataT), FinalLambda>(
      x, y, m, n, k, dOutput, fin_op, stream);
  } else {
    l1Impl<DataT, AccT, OutT, IdxT, 1, FinalLambda>(x, y, m, n, k, dOutput,
                                                    fin_op, stream);
  }
}

/**
 * @brief the L1 distance matrix calculation
 *  It computes the following equation: cij = op(ai-bj)
 * @tparam InType input data-type (for A and B matrices)
 * @tparam AccType accumulation data-type
 * @tparam OutType output data-type (for C and D matrices)
 * @tparam OutputTile_ output tile size for the thread block
 * @tparam FinalLambda user-defined epilogue lamba
 * @tparam Index_ Index type
 * @param m number of rows of A and C/D
 * @param n number of columns of B and C/D
 * @param k number of cols of A and rows of B
 * @param pA input matrix
 * @param pB input matrix
 * @param pD output matrix
 * @param fin_op the final element-wise epilogue lambda
 * @param stream cuda stream where to launch work
 * @param isRowMajor whether the input and output matrices are row major
 */
template <typename InType, typename AccType, typename OutType,
          typename OutputTile_, typename FinalLambda, typename Index_ = int>
void l1Impl(int m, int n, int k, const InType *pA, const InType *pB,
            OutType *pD, FinalLambda fin_op, cudaStream_t stream,
            bool isRowMajor) {
  typedef std::is_same<OutType, bool> is_bool;

  if (isRowMajor) {
    typedef typename std::conditional<is_bool::value, OutType, AccType>::type
      L1OutType;
    l1<InType, AccType, L1OutType, Index_, FinalLambda>(
      m, n, k, pA, pB, reinterpret_cast<L1OutType *>(pD), fin_op, stream);

  } else {
    typedef typename std::conditional<is_bool::value, AccType, OutType>::type
      EffOutType;
    EffOutType *pDCast =
      reinterpret_cast<EffOutType *>(pD);  // Pretend to be EffOutType;

    typedef cutlass::Shape<8, 8, 8> AccumulatorsPerThread_;
    typedef LinAlg::ThreadL1NormAdd<
      AccumulatorsPerThread_, cutlass::Shape<1, 4, 8>, InType, InType, AccType>
      MainLoopFunctor_;
    typedef LinAlg::CustomGemmConfig<InType, AccType, EffOutType, OutputTile_,
                                     AccumulatorsPerThread_, MainLoopFunctor_>
      GemmConfig_;

    typedef UnexpandedDistanceFragmentMultiplyAdd FragmentMultiplyAdd_;

    typedef UnexpandedDistanceEpilogueFunctor<EffOutType, GemmConfig_,
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
    typedef UnexpandedDistanceGemmEpilogue<GemmEpilogueTraits_> GemmEpilogue_;
    typedef typename EpilogueFunctor_::Params EpiParams;

    cublasOperation_t transa, transb;
    const InType *aPtr, *bPtr;
    Index_ lda, ldb, ldd;
    Index_ gemm_m, gemm_n;
    transa = CUBLAS_OP_N;
    transb = CUBLAS_OP_T;
    aPtr = pA;
    bPtr = pB;
    lda = m;
    ldb = n;
    ldd = m;
    gemm_m = m;
    gemm_n = n;

    LinAlg::gemm<InType, AccType, EffOutType, OutputTile_,
                 AccumulatorsPerThread_, MainLoopFunctor_, Index_, GemmConfig_,
                 EpilogueFunctor_, GemmEpilogueTraits_, GemmEpilogue_>(
      transa, transb, gemm_m, gemm_n, k, (EffOutType)1, aPtr, lda, bPtr, ldb,
      (EffOutType)0, nullptr, ldd, pDCast,
      [] HD(EpiParams & p) {
        int err = p.initializeExtra(nullptr, nullptr, false);
        return err;
      },
      fin_op, stream);
  }
}
}  // namespace Distance
}  // namespace MLCommon

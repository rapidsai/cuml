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

#include <cublas_v2.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/gemm_epilogue.h>
#include <cutlass/gemm/gemm_epilogue_traits.h>
#include <cutlass/gemm/gemm_traits.h>
#include <cutlass/gemm/linear_scaling.h>
#include <cutlass/gemm/thread_multiply_add.h>
#include <cutlass/fragment_multiply_add.h>
#include <cutlass/coord.h>
#include <cutlass/util/platform.h>
#include "cublas_wrappers.h"
#include "cuda_utils.h"


namespace MLCommon {
namespace LinAlg {


/**
 * this type has been mostly customized for float/double data-types
 * might require changes to the template params for others!
 */
template <
  /// Input type
  typename IType,
  /// accumulator type
  typename AccType_,
  /// Output type
  typename OType,
  /// The tile size for the GEMM KxNxM.
  typename OutputTile_,
  /// The number of accumulators per thread.
  typename AccumulatorsPerThread_,
  /// main loop functor
  typename MainLoopFunctor_,
  /// The number of scalars per LDG for A.
  int kScalarsPerLdgA_ = 1,
  /// The number of scalars per LDG for B.
  int kScalarsPerLdgB_ = 1,
  /// The number of scalars per LDG for Acc.
  int kScalarsPerLdgAcc_ = 1,
  /// The number of scalars per LDG/STG/LDS for C or D.
  int kScalarsPerLdgC_ = 1>
struct CustomGemmConfig
  : public cutlass::gemm::GemmConfig<
      /// The scalar type for A.
      IType,
      /// The scalar type for B.
      IType,
      /// The scalar type for C.
      OType,
      /// The scalar type for D.
      OType,
      /// The tile size for the GEMM KxNxM.
      OutputTile_,
      /// The functor to do the math in the main loop.
      MainLoopFunctor_,
      /// The number of scalars per LDG for A.
      kScalarsPerLdgA_,
      /// The number of scalars per STS for A.
      kScalarsPerLdgA_,
      /// The number of scalars per LDS for A.
      16 / sizeof(IType),
      /// The number of scalars per LDG for B.
      kScalarsPerLdgB_,
      /// The number of scalars per STS for B.
      kScalarsPerLdgB_,
      /// The number of scalars per LDS for B.
      16 / sizeof(IType),
      /// The number of scalars per LDG for C and STG for D.
      kScalarsPerLdgC_,
      /// The number of scalars per STS for D.
      16 / sizeof(OType),
      /// The number of scalars per LDS for D.
      kScalarsPerLdgC_,
      /// The number of stages in shared memory.
      2> {
    /// Acc Type
    typedef AccType_ AccType;
    /// number of scalars per LDG for Acc
    enum { kScalarsPerLdgAcc = kScalarsPerLdgAcc_ };
    /// number of scalars per STS for Acc
    enum { kScalarsPerStsAcc = kScalarsPerLdgAcc_ };
    /// number of scalars per LDS for Acc
    enum { kScalarsPerLdsAcc = 16 / sizeof(AccType) };
};


/***
 * Exposes passing of a different type in which to accumulate in epilogue
 */
template <
    /// The GEMM config
    typename GemmConfig_,
    /// functor ot use in the epilogue
    typename EpilogueFunctor_,
    /// index
    typename Index_,
    typename BaseClass =
    cutlass::gemm::SimplifiedGemmEpilogueTraits<
        GemmConfig_, EpilogueFunctor_, Index_>>
struct CustomGemmEpilogueTraits : public BaseClass {
    /// for passing accumulator type itself directly to epilogue functor!
    typedef typename GemmConfig_::AccType AccType;

    /// traits class to build the iterator to store to shared memory for Acc
    typedef cutlass::gemm::GemmSharedStoreTileDTraits<
        // The pointer is float.
        AccType,
        // The output tile size.
        typename GemmConfig_::OutputTile,
        // The number of warps.
        typename GemmConfig_::Warps,
        // The number of threads per warp.
        typename GemmConfig_::MultiplyAdd::ThreadsPerWarp,
        // The number of scalars per STS.
        GemmConfig_::kScalarsPerStsAcc,
        // The skew -- 128 / sizeof(AccType) / kScalarsPerStsB is the number of
        // threads involved in a single STS. We divide by 2 as our objective is
        // to add a skew to the odd threads to avoid bank conflicts between odd
        // and even threads.
        // it is assumed here that InType == AccType!
        128 / sizeof(AccType) / GemmConfig_::kScalarsPerStsB / 2 *
        GemmConfig_::kScalarsPerStsB>
    SharedStoreTileTraitsAcc;

    /// The iterator to store D to shared memory.
    typedef cutlass::TileStoreIterator<SharedStoreTileTraitsAcc,
                                       typename SharedStoreTileTraitsAcc::Scalar,
                                       cutlass::IteratorAdvance::kH,
                                       cutlass::MemorySpace::kShared>
    SharedStoreIteratorAcc;

    /// The shared store transformer for Acc
    typedef cutlass::Copy<typename SharedStoreIteratorAcc::Fragment>
    SharedStoreTransformerAcc;

    /// The traits class to build the iterator to load from shared memory for Acc.
    typedef cutlass::gemm::GemmSharedLoadTileDTraits<
        // The pointer is float.
        AccType,
        // The output tile size.
        typename GemmConfig_::OutputTile,
        // The number of warps.
        typename GemmConfig_::Warps,
        // The number of threads per warp.
        typename GemmConfig_::MultiplyAdd::ThreadsPerWarp,
        // The number of columns of the output tile written by iteration.
        GemmConfig_::OutputTile::kH /
          cutlass::ShapeCount<typename BaseClass::Iterations>::kCount,
        // The number of scalars per LDS.
        GemmConfig_::kScalarsPerLdsB,
        // The skew.
        SharedStoreTileTraitsAcc::kSkew>
    SharedLoadTileTraitsAcc;

    /// The iterator to load Acc from shared memory.
    typedef cutlass::TileLoadIterator<SharedLoadTileTraitsAcc,
                                      typename SharedLoadTileTraitsAcc::Scalar,
                                      cutlass::IteratorAdvance::kH,
                                      cutlass::MemorySpace::kShared>
    SharedLoadIteratorAcc;
}; // end struct CustomGemmEpilogueTraits


template <typename GemmEpilogueTraits_,
          typename BaseClass = cutlass::gemm::GemmEpilogue<GemmEpilogueTraits_>>
struct CustomGemmEpilogue : public BaseClass {
  /// The traits class.
  typedef GemmEpilogueTraits_ Traits;
  using typename BaseClass::Params;
  using typename BaseClass::SharedStorage;
  using typename BaseClass::Index;
  using typename BaseClass::Accumulators;

  /// Ctor.
  CUTLASS_DEVICE CustomGemmEpilogue(Params const& params_,
                                    SharedStorage& shared_storage_,
                                    Index m_, Index n_)
    : BaseClass(params_, shared_storage_, m_, n_) {}

  /// Execute the epilogue.
  template <typename FinalLambda>
  CUTLASS_DEVICE void epilogue(cutlass::Coord<3> const& block,
                               Accumulators& accumulators, FinalLambda fin_op) {
    BaseClass::epilogue(block, accumulators);
  }
}; // end struct CustomGemmEpilogue


template <typename Scalar_,
          typename FragmentMultiplyAdd_ = cutlass::gemm::FragmentMultiplyAdd<Scalar_>,
          typename BaseClass = cutlass::gemm::LinearScaling<
            Scalar_, FragmentMultiplyAdd_>>
struct LinearScaling : public BaseClass {
  using typename BaseClass::Params;

  CUTLASS_DEVICE LinearScaling(Params const& params) : BaseClass(params) {}

  template <typename FragmentA_, typename FragmentB_>
  CUTLASS_DEVICE void evaluate(FragmentA_ const& accum, FragmentB_& output) {
    BaseClass::evaluate(accum, output);
  }

  template <typename FragmentA_, typename FragmentB_>
  CUTLASS_DEVICE void evaluate(FragmentA_ const& accum, FragmentB_ const& old,
                               FragmentB_& output) {
    BaseClass::evaluate(accum, old, output);
  }

  template <typename FragmentA_, typename FragmentB_, typename FinalLambda>
  CUTLASS_DEVICE void evaluate(FragmentA_ const& accum, FragmentB_& output,
                               FinalLambda fin_op) {
    BaseClass::evaluate(accum, output);
  }

  template <typename FragmentA_, typename FragmentB_, typename FinalLambda>
  CUTLASS_DEVICE void evaluate(FragmentA_ const& accum, FragmentB_ const& old,
                               FragmentB_& output, FinalLambda fin_op) {
    BaseClass::evaluate(accum, old, output);
  }
};


/**
 * main traits to customize cutlass gemm kernel
 * this type has been mostly customized for float/double data-types
 * might require changes to the template params for others!
 */
template <
  /// Input type
  typename IType,
  /// Accumulation type
  typename AccType,
  /// Output type
  typename OType,
  /// The layout for A.
  cutlass::MatrixLayout::Kind kLayoutA_,
  /// The layout for B.
  cutlass::MatrixLayout::Kind kLayoutB_,
  /// The output tile.
  typename OutputTile_,
  /// The number of accumulators per thread.
  typename AccumulatorsPerThread_ = cutlass::Shape<8, 8, 8>,
  /// the functor to use in main loop
  typename MainLoopFunctor_ = cutlass::gemm::ThreadMultiplyAdd<
    AccumulatorsPerThread_, cutlass::Shape<1, 4, 8>, IType, IType, AccType>,
  /// The index.
  typename Index_ = int,
  /// The GEMM config.
  typename GemmConfig_ = CustomGemmConfig<
    IType, AccType, OType, OutputTile_, AccumulatorsPerThread_,
    MainLoopFunctor_>,
  /// The functor to use in the epilogue.
  typename EpilogueFunctor_ = LinearScaling<OType>,
  /// The traits class for the epilogue.
  typename GemmEpilogueTraits_ = CustomGemmEpilogueTraits<
    GemmConfig_, EpilogueFunctor_, Index_>,
  /// The class for the epilogue.
  typename GemmEpilogue_ = CustomGemmEpilogue<GemmEpilogueTraits_>>
struct CustomGemmTraits : public cutlass::gemm::SimplifiedGemmTraits<
                            // The layout for A.
                            kLayoutA_,
                            // The layout for B.
                            kLayoutB_,
                            // The config.
                            GemmConfig_,
                            // The epilogue.
                            GemmEpilogue_,
                            // The index.
                            Index_> {};


template <typename Gemm_, typename FinalLambda>
__global__ void custom_gemm_kernel(typename Gemm_::Params params,
                                   FinalLambda fin_op) {
  __shared__ typename Gemm_::SharedStorage shared_storage;
  Gemm_ gemm(params, shared_storage);
  gemm.multiply_add(fin_op);
}

/**
 * main Gemm class to launch the kernel. It is customized to accept a device
 * lambda directly. This is done this way since cutlass currently doesn't expose
 * such an interface.
 */
template <typename GemmTraits_,
          typename BaseClass = cutlass::gemm::Gemm<GemmTraits_>>
struct CustomGemm : public BaseClass {
  /// This class.
  typedef CustomGemm<GemmTraits_> This_;
  /// The traits.
  typedef typename BaseClass::Traits Traits;
  /// The shared storage.
  typedef typename Traits::SharedStorage SharedStorage;

  /// The scalar for A.
  typedef typename Traits::ScalarA ScalarA;
  /// The scalar for B.
  typedef typename Traits::ScalarB ScalarB;
  /// The scalar in the epilogue.
  typedef typename Traits::Epilogue::Scalar ScalarEpilogue;
  /// The scalar for C.
  typedef typename Traits::Epilogue::ScalarC ScalarC;
  /// The scalar for D.
  typedef typename Traits::Epilogue::ScalarD ScalarD;
  /// The index.
  typedef typename Traits::Index Index;

  /// params
  struct Params : public BaseClass::Params {
    CUTLASS_HOST_DEVICE int initialize(Index m,
                                       Index n,
                                       Index k,
                                       ScalarEpilogue alpha,
                                       void const* d_a,
                                       Index lda,
                                       void const* d_b,
                                       Index ldb,
                                       ScalarEpilogue beta,
                                       void const* d_c,
                                       Index ldc,
                                       void* d_d,
                                       Index ldd) {
      cutlass::gemm::GemmDesc<ScalarEpilogue, Index> desc;
      desc.m = m;
      desc.n = n;
      desc.k = k;
      desc.alpha = alpha;
      desc.beta = beta;
      desc.d_a = d_a;
      desc.lda = lda;
      desc.d_b = d_b;
      desc.ldb = ldb;
      desc.d_c = d_c;
      desc.ldc = ldc;
      desc.d_d = d_d;
      desc.ldd = ldd;
      return Traits::Params::initialize(desc);
    }
  };

  /// Launch the kernel.
  template <typename FinalLambda>
  static void launch(Params const& params, FinalLambda fin_op,
                     cudaStream_t stream) {
    // Setup the grid.
    dim3 grid;
    grid.x = ceildiv(params.m, Traits::OutputTile::kW);
    grid.y = ceildiv(params.n, Traits::OutputTile::kH);
    // The number of threads.
    dim3 block;
    block.x = BaseClass::kThreads;
    // Launch the kernel.
    void const* args[] = {&params, &fin_op};
    cudaLaunchKernel(reinterpret_cast<void*>(&custom_gemm_kernel<This_, FinalLambda>),
                     grid,
                     block,
                     const_cast<void**>(args),
                     0,
                     stream);
    CUDA_CHECK(cudaPeekAtLastError());
  }

  static void launch(Params const& params, cudaStream_t stream) {
    BaseClass::launch(params, stream);
  }

  /// Ctor.
  CUTLASS_DEVICE CustomGemm(Params const& params_,
                            SharedStorage& shared_storage_)
    : BaseClass(params_, shared_storage_) {}

  /// Do the GEMM.
  template <typename FinalLambda>
  CUTLASS_DEVICE void multiply_add(FinalLambda fin_op) {
    // Swizzle the IDs of the block (to enable better cache behavior).
    typename Traits::BlockSwizzle block_swizzle;
    dim3 block = block_swizzle.swizzle();

    // Scale the id.
    block.x *= Traits::OutputTile::kW;
    block.y *= Traits::OutputTile::kH;

    // We may want to use shared memory to clear the registers.
    typedef typename Traits::ClearAccumulators ClearAccumulators;

    // The streams to read A/B from global memory to shared memory.
    typename Traits::GlobalLoadStream global_stream(BaseClass::params,
                                                    BaseClass::shared_storage,
                                                    block);

    // Create the accumulator clear.
    ClearAccumulators clear(BaseClass::shared_storage.main_loop.clear);

    // By how much we unroll the main loop.
    Index const kUnroll = static_cast<Index>(Traits::OutputTile::kD);

    // If we do not have enough steps in the main loop, trigger the residue code
    global_stream.move_to_residue<true>(BaseClass::params.k);

    // Fetch the fragments for A and B from global memory.
    global_stream.copy();

    // Copy the elements to shared memory (after transformation if needed).
    global_stream.commit();

    // Make sure the data is in shared memory.
    Traits::shared_store_fence(false);

    // Rollback to the beginning of the GEMM-K dimension. It may have no impact.
    global_stream.rollback();

    // The unrolling steps for the main loop.
    int const kUnrollingSteps =
      Traits::MultiplyAdd::AccumulatorsPerWarp::kD /
      Traits::MultiplyAdd::InstructionShape::kD;

    // Make sure we have at least 2 unrolling steps or our pipeling is not
    // going to work.
    static_assert(kUnrollingSteps >= 2,
                  "The pipelining assumes at least two steps");

    // The stream of data from shared memory to fragments.
    typename Traits::SharedLoadStream shared_load_stream(
      BaseClass::params, BaseClass::shared_storage);

    // Trigger the copy from shared memory for the 1st stream.
    shared_load_stream.copy(0);

    // Allocate the accumulators.
    typename Traits::MultiplyAdd::Accumulators accumulators;
    // Clear the accumulators.
    clear.clear(accumulators);

    // The loop index.
    Index outer_k = BaseClass::params.k - kUnroll;

    // Enter the main loop and iterate.
    for (; outer_k > 0; outer_k -= kUnroll) {
      BaseClass::consume_tile<false>(global_stream, shared_load_stream,
                                     accumulators, outer_k);
    }

    // Residual loop.
    for (; outer_k > -kUnroll; outer_k -= kUnroll) {
      BaseClass::consume_tile<true>(global_stream, shared_load_stream,
                                    accumulators, outer_k);
    }

    // Epilogue.
    typedef typename Traits::Epilogue Epilogue;
    Epilogue epilogue(BaseClass::params.epilogue,
                      BaseClass::shared_storage.epilogue, BaseClass::params.m,
                      BaseClass::params.n);
    epilogue.epilogue(cutlass::make_Coord(0, block.y, block.x), accumulators,
                      fin_op);
  }
}; // end struct CustomGemm


/**
 * @brief main function to launch cutlass-gemm kernel. It computes the following
 *  equation: D = alpha . opA(A) * opB(B) + beta . C
 * @tparam IType input data-type (for A and B matrices)
 * @tparam AccType accumulation data-type
 * @tparam OType output data-type (for C and D matrices)
 * @tparam kLayoutA layout for A
 * @tparam kLayoutB layout for B
 * @tparam OutputTile_ output tile size for the thread block
 * @tparam AccumulatorsPerThread_ number of accumulators per thread
 * @tparam MainLoopFunctor_ custom functor to be used in the main loop
 * @tparam Index_ the type of index
 * @tparam GemmConfig_ the config for the GEMM
 * @tparam EpilogueFunctor_ custom epilogue functor
 * @tparam GemmEpilogueTraits_ epilogue traits class to build the epilogue
 * @tparam GemmEpilogue_ custom epilogue
 * @tparam Lambda lambda to initialize any custom params inside EpilogueFunctor_
 * @tparam FinalLambda Final device lambda to be applied in epilogue
 * @param transA cublas transpose op for A
 * @param transB cublas transpose op for B
 * @param m number of rows of A and C/D
 * @param n number of columns of B and C/D
 * @param k number of cols of A and rows of B
 * @param alpha scalar
 * @param A input matrix
 * @param lda leading dim for A
 * @param B input matrix
 * @param ldb leading dim for B
 * @param beta scalar
 * @param C input matrix
 * @param ldc leading dim for C and D
 * @param D output matrix
 * @param op lambda function to initialize any custom params inside
 * EpilogueFunctor_
 * @param fin_op the final lambda to be run inside the Epilogue. This can help
 * in customizing a given EpilogueFunctor, without having to go through the task
 * of creating another Functor!
 * @param stream cuda stream where to launch work
 *
 * @note op: This is a host-side lambda to initialize any custom params needed
 * by EpilogueFunctor_. It's signature is as follows:
 * <pre>void op(EpilogueFunctor_& func);</pre>
 *
 * @note fin_op: This is a device-side lambda to perform an elementwise op on
 * the accumulated result and return the result. It's signature is as follows:
 * <pre>OType fin_op(AccType val, int g_idx);</pre>
 * @{
 */
template <
  typename IType, typename AccType, typename OType,
  cutlass::MatrixLayout::Kind kLayoutA, cutlass::MatrixLayout::Kind kLayoutB,
  typename OutputTile_,
  typename AccumulatorsPerThread_ = cutlass::Shape<8, 8, 8>,
  typename MainLoopFunctor_ = cutlass::gemm::ThreadMultiplyAdd<
    AccumulatorsPerThread_, cutlass::Shape<1, 4, 8>, IType, IType, AccType>,
  typename Index_ = int,
  typename GemmConfig_ = CustomGemmConfig<
    IType, AccType, OType, OutputTile_, AccumulatorsPerThread_,
    MainLoopFunctor_>,
  typename EpilogueFunctor_ = LinearScaling<OType>,
  typename GemmEpilogueTraits_ = CustomGemmEpilogueTraits<
    GemmConfig_, EpilogueFunctor_, Index_>,
  typename GemmEpilogue_ = CustomGemmEpilogue<GemmEpilogueTraits_>,
  typename Lambda, typename FinalLambda>
void gemmLauncher(cublasOperation_t transA, cublasOperation_t transB, int m,
                  int n, int k, OType alpha, IType const *A, int lda,
                  IType const *B, int ldb, OType beta, OType const *C, int ldc,
                  OType *D, Lambda op, FinalLambda fin_op,
                  cudaStream_t stream) {
  typedef CustomGemmTraits<IType, AccType, OType, kLayoutA, kLayoutB,
                           OutputTile_, AccumulatorsPerThread_,
                           MainLoopFunctor_, Index_, GemmConfig_,
                           EpilogueFunctor_, GemmEpilogueTraits_, GemmEpilogue_>
    GemmTraits;
  typedef CustomGemm<GemmTraits> Gemm;
  typename Gemm::Params params;
  int err =
    params.initialize(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D, ldc);
  ASSERT(err == 0, "gemmLauncher: params.initialize failed err=%d", err);
  err = op(params.epilogue.functor);
  ASSERT(err == 0, "gemmLauncher: op(epiloguefunctor) failed err=%d", err);
  Gemm::launch(params, fin_op, stream);
}

template <
  typename IType, typename AccType, typename OType,
  cutlass::MatrixLayout::Kind kLayoutA, cutlass::MatrixLayout::Kind kLayoutB,
  typename OutputTile_,
  typename AccumulatorsPerThread_ = cutlass::Shape<8, 8, 8>,
  typename MainLoopFunctor_ = cutlass::gemm::ThreadMultiplyAdd<
    AccumulatorsPerThread_, cutlass::Shape<1, 4, 8>, IType, IType, AccType>,
  typename Index_ = int,
  typename GemmConfig_ = CustomGemmConfig<
    IType, AccType, OType, OutputTile_, AccumulatorsPerThread_,
    MainLoopFunctor_>,
  typename EpilogueFunctor_ = cutlass::gemm::LinearScaling<OType>,
  typename GemmEpilogueTraits_ = CustomGemmEpilogueTraits<
    GemmConfig_, EpilogueFunctor_, Index_>,
  typename GemmEpilogue_ = cutlass::gemm::GemmEpilogue<GemmEpilogueTraits_>,
  typename Lambda>
void gemmLauncher(cublasOperation_t transA, cublasOperation_t transB, int m,
                  int n, int k, OType alpha, IType const *A, int lda,
                  IType const *B, int ldb, OType beta, OType const *C, int ldc,
                  OType *D, Lambda op, cudaStream_t stream) {
  typedef CustomGemmTraits<IType, AccType, OType, kLayoutA, kLayoutB,
                           OutputTile_, AccumulatorsPerThread_,
                           MainLoopFunctor_, Index_, GemmConfig_,
                           EpilogueFunctor_, GemmEpilogueTraits_, GemmEpilogue_>
    GemmTraits;
  typedef cutlass::gemm::Gemm<GemmTraits> Gemm;
  typename Gemm::Params params;
  int err =
    params.initialize(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D, ldc);
  ASSERT(err == 0, "gemmLauncher: params.initialize failed err=%d", err);
  err = op(params.epilogue.functor);
  ASSERT(err == 0, "gemmLauncher: op(epiloguefunctor) failed err=%d", err);
  Gemm::launch(params, stream);
}
/** @} */


/**
 * @brief the wrapper of gemmLauncher, which doesn't need to specify
 *  cutlass::MatrixLayout::Kind. It computes the following equation:
 *  D = alpha . opA(A) * opB(B) + beta . C
 * @tparam IType input data-type (for A and B matrices)
 * @tparam AccType accumulation data-type
 * @tparam OType output data-type (for C and D matrices)
 * @tparam OutputTile_ output tile size for the thread block
 * @tparam AccumulatorsPerThread_ number of accumulators per thread
 * @tparam MainLoopFunctor_ custom functor to be used in the main loop
 * @tparam Index_ the type of index
 * @tparam GemmConfig_ the config for the GEMM
 * @tparam EpilogueFunctor_ custom epilogue functor
 * @tparam GemmEpilogueTraits_ epilogue traits class to build the epilogue
 * @tparam GemmEpilogue_ custom epilogue
 * @tparam Lambda lambda to initialize any custom params inside EpilogueFunctor_
 * @tparam FinalLambda Final device lambda to be applied in epilogue
 * @param transA cublas transpose op for A
 * @param transB cublas transpose op for B
 * @param m number of rows of A and C/D
 * @param n number of columns of B and C/D
 * @param k number of cols of A and rows of B
 * @param alpha scalar
 * @param A input matrix
 * @param lda leading dim for A
 * @param B input matrix
 * @param ldb leading dim for B
 * @param beta scalar
 * @param C input matrix
 * @param ldc leading dim for C and D
 * @param D output matrix
 * @param op lambda function to initialize any custom params inside
 * EpilogueFunctor_
 * @param fin_op the final lambda to be run inside the Epilogue. This can help
 * in customizing a given EpilogueFunctor, without having to go through the task
 * of creating another Functor!
 * @param stream cuda stream where to launch work
 * @{
 */
template <
  typename IType, typename AccType, typename OType, typename OutputTile_,
  typename AccumulatorsPerThread_ = cutlass::Shape<8, 8, 8>,
  typename MainLoopFunctor_ = cutlass::gemm::ThreadMultiplyAdd<
    AccumulatorsPerThread_, cutlass::Shape<1, 4, 8>, IType, IType, AccType>,
  typename Index_ = int,
  typename GemmConfig_ = CustomGemmConfig<
    IType, AccType, OType, OutputTile_, AccumulatorsPerThread_,
    MainLoopFunctor_>,
  typename EpilogueFunctor_ = LinearScaling<OType>,
  typename GemmEpilogueTraits_ = CustomGemmEpilogueTraits<
    GemmConfig_, EpilogueFunctor_, Index_>,
  typename GemmEpilogue_ = CustomGemmEpilogue<GemmEpilogueTraits_>,
  typename Lambda, typename FinalLambda>
void baseGemm(cublasOperation_t transA, cublasOperation_t transB, int m, int n,
              int k, OType alpha, IType const *A, int lda, IType const *B,
              int ldb, OType beta, OType const *C, int ldc, OType *D, Lambda op,
              FinalLambda fin_op, cudaStream_t stream) {
  if (transA == CUBLAS_OP_N && transB == CUBLAS_OP_N) {
    gemmLauncher<IType, AccType, OType, cutlass::MatrixLayout::kColumnMajor,
                 cutlass::MatrixLayout::kColumnMajor, OutputTile_,
                 AccumulatorsPerThread_, MainLoopFunctor_, Index_, GemmConfig_,
                 EpilogueFunctor_, GemmEpilogueTraits_, GemmEpilogue_>(
      transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D, op,
      fin_op, stream);
  } else if (transA == CUBLAS_OP_N && transB == CUBLAS_OP_T) {
    gemmLauncher<IType, AccType, OType, cutlass::MatrixLayout::kColumnMajor,
                 cutlass::MatrixLayout::kRowMajor, OutputTile_,
                 AccumulatorsPerThread_, MainLoopFunctor_, Index_, GemmConfig_,
                 EpilogueFunctor_, GemmEpilogueTraits_, GemmEpilogue_>(
      transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D, op,
      fin_op, stream);
  } else if (transA == CUBLAS_OP_T && transB == CUBLAS_OP_N) {
    gemmLauncher<IType, AccType, OType, cutlass::MatrixLayout::kRowMajor,
                 cutlass::MatrixLayout::kColumnMajor, OutputTile_,
                 AccumulatorsPerThread_, MainLoopFunctor_, Index_, GemmConfig_,
                 EpilogueFunctor_, GemmEpilogueTraits_, GemmEpilogue_>(
      transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D, op,
      fin_op, stream);
  } else if (transA == CUBLAS_OP_T && transB == CUBLAS_OP_T) {
    gemmLauncher<IType, AccType, OType, cutlass::MatrixLayout::kRowMajor,
                 cutlass::MatrixLayout::kRowMajor, OutputTile_,
                 AccumulatorsPerThread_, MainLoopFunctor_, Index_, GemmConfig_,
                 EpilogueFunctor_, GemmEpilogueTraits_, GemmEpilogue_>(
      transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D, op,
      fin_op, stream);
  } else {
    ASSERT(false, "runGemm: Bad cublasOperation_t a=%d b=%d\n", (int)transA,
           (int)transB);
  }
  CUDA_CHECK(cudaPeekAtLastError());
}

template <
  typename IType, typename AccType, typename OType, typename OutputTile_,
  typename AccumulatorsPerThread_ = cutlass::Shape<8, 8, 8>,
  typename MainLoopFunctor_ = cutlass::gemm::ThreadMultiplyAdd<
    AccumulatorsPerThread_, cutlass::Shape<1, 4, 8>, IType, IType, AccType>,
  typename Index_ = int,
  typename GemmConfig_ = CustomGemmConfig<
    IType, AccType, OType, OutputTile_, AccumulatorsPerThread_,
    MainLoopFunctor_>,
  typename EpilogueFunctor_ = cutlass::gemm::LinearScaling<OType>,
  typename GemmEpilogueTraits_ = CustomGemmEpilogueTraits<
    GemmConfig_, EpilogueFunctor_, Index_>,
  typename GemmEpilogue_ = cutlass::gemm::GemmEpilogue<GemmEpilogueTraits_>,
  typename Lambda>
void baseGemm(cublasOperation_t transA, cublasOperation_t transB, int m, int n,
              int k, OType alpha, IType const *A, int lda, IType const *B,
              int ldb, OType beta, OType const *C, int ldc, OType *D, Lambda op,
              cudaStream_t stream) {
  if (transA == CUBLAS_OP_N && transB == CUBLAS_OP_N) {
    gemmLauncher<IType, AccType, OType, cutlass::MatrixLayout::kColumnMajor,
                 cutlass::MatrixLayout::kColumnMajor, OutputTile_,
                 AccumulatorsPerThread_, MainLoopFunctor_, Index_, GemmConfig_,
                 EpilogueFunctor_, GemmEpilogueTraits_, GemmEpilogue_>(
      transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D, op,
      stream);
  } else if (transA == CUBLAS_OP_N && transB == CUBLAS_OP_T) {
    gemmLauncher<IType, AccType, OType, cutlass::MatrixLayout::kColumnMajor,
                 cutlass::MatrixLayout::kRowMajor, OutputTile_,
                 AccumulatorsPerThread_, MainLoopFunctor_, Index_, GemmConfig_,
                 EpilogueFunctor_, GemmEpilogueTraits_, GemmEpilogue_>(
      transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D, op,
      stream);
  } else if (transA == CUBLAS_OP_T && transB == CUBLAS_OP_N) {
    gemmLauncher<IType, AccType, OType, cutlass::MatrixLayout::kRowMajor,
                 cutlass::MatrixLayout::kColumnMajor, OutputTile_,
                 AccumulatorsPerThread_, MainLoopFunctor_, Index_, GemmConfig_,
                 EpilogueFunctor_, GemmEpilogueTraits_, GemmEpilogue_>(
      transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D, op,
      stream);
  } else if (transA == CUBLAS_OP_T && transB == CUBLAS_OP_T) {
    gemmLauncher<IType, AccType, OType, cutlass::MatrixLayout::kRowMajor,
                 cutlass::MatrixLayout::kRowMajor, OutputTile_,
                 AccumulatorsPerThread_, MainLoopFunctor_, Index_, GemmConfig_,
                 EpilogueFunctor_, GemmEpilogueTraits_, GemmEpilogue_>(
      transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D, op,
      stream);
  } else {
    ASSERT(false, "runGemm: Bad cublasOperation_t a=%d b=%d\n", (int)transA,
           (int)transB);
  }
  CUDA_CHECK(cudaPeekAtLastError());
}
/** @} */

}; // end namespace LinAlg
}; // end namespace MLCommon

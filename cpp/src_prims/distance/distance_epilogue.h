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

#include <cutlass/convert.h>
#include <cutlass/coord.h>
#include <cutlass/fragment.h>
#include <cutlass/gemm/gemm_epilogue.h>
#include <cutlass/gemm/gemm_global_tile.h>
#include <cutlass/gemm/gemm_traits.h>
#include <cutlass/iterator_access.h>
#include <cutlass/shape.h>

namespace MLCommon {
namespace Distance {

namespace {
template <typename OutputIterator>
CUTLASS_HOST_DEVICE void
  extract_index_from_iterator(OutputIterator &iterator,
                              typename OutputIterator::Pointer base_ptr,
                              int index[OutputIterator::Fragment::kElements]) {
  int st = 0;
  typename OutputIterator::Pointer current_ptr = iterator.params.pointer;
  typename OutputIterator::Index current_pred_offset =
    iterator.params.predicate_offset;
  for (int d = 0; d < OutputIterator::Iterations::kD; ++d) {
    for (int h = 0; h < OutputIterator::Iterations::kH; ++h) {
      for (int w = 0; w < OutputIterator::Iterations::kW; ++w) {
        int const imm = cutlass::ComputeOffsetFromStrides<
          typename OutputIterator::Base::ImmediateOffsetStrides>::get(0, 0, w,
                                                                      0);
        index[st++] = iterator.valid(d, h, w, 0)
                        ? (&iterator.params.pointer[imm] - base_ptr)
                        : -1;
        if (w < OutputIterator::Iterations::kW - 1) {
          iterator.inc_w();
        }
      }
      if (h < OutputIterator::Iterations::kH - 1) {
        iterator.inc_h();
      }
    }
    if (d < OutputIterator::Iterations::kD - 1) {
      iterator.inc_d();
    }
  }
  iterator.inc_advance();
  iterator.params.pointer = current_ptr;
  iterator.params.predicate_offset = current_pred_offset;
}

} // end anonymous namespace



/**
 * @brief Base Epilogue for distance metrics
 * @tparam GemmEpilogueTraits_ the traits class to configure this epilogue
 */
template <typename GemmEpilogueTraits_>
struct DistanceGemmEpilogue {
  /// The traits class.
  typedef GemmEpilogueTraits_ Traits;
  /// The params.
  typedef typename Traits::Params Params;
  /// The shared storage.
  typedef typename Traits::SharedStorage SharedStorage;

  /// The output tile.
  typedef typename Traits::OutputTile OutputTile;
  /// The number of iterations.
  typedef typename Traits::Iterations Iterations;
  /// The accumulators.
  typedef typename Traits::Accumulators Accumulators;
  /// The scalar.
  typedef typename Traits::Scalar Scalar;
  /// The functor in charge of the math.
  typedef typename Traits::Functor Functor;

  /// We do not support 3D or 4D shapes.
  static_assert(Iterations::kD == 1 && Iterations::kC == 1,
                "Unsupported 3D/4D shapes");

  /// The iterator for C in global memory.
  typedef typename Traits::GlobalLoadIteratorC GlobalLoadIteratorC;
  /// The transformer for C.
  typedef typename Traits::GlobalTransformerC GlobalTransformerC;
  /// The transformer for D.
  typedef typename Traits::GlobalTransformerD GlobalTransformerD;
  /// The iterator for D in global memory.
  typedef typename Traits::GlobalStoreIteratorD GlobalStoreIteratorD;
  /// The iterator to store D in shared memory.
  typedef typename Traits::SharedStoreIteratorD SharedStoreIteratorD;
  /// The shared store transformer for D.
  typedef typename Traits::SharedStoreTransformerD SharedStoreTransformerD;
  /// The iterator to load D in shared memory.
  typedef typename Traits::SharedLoadIteratorD SharedLoadIteratorD;
  /// The shared load transformer for D.
  typedef cutlass::Copy<typename SharedLoadIteratorD::Fragment>
    SharedLoadTransformerD;

  /// The index.
  typedef typename Traits::Index Index;

  /// The scalar for C.
  typedef typename GlobalLoadIteratorC::Scalar ScalarC;
  /// The scalar for D.
  typedef typename GlobalStoreIteratorD::Scalar ScalarD;

  // The AA fragment
  typedef typename Functor::GlobalLoadIteratorAA GlobalLoadIteratorAA;
  // The BB fragment
  typedef typename Functor::GlobalLoadIteratorBB GlobalLoadIteratorBB;

  /// Ctor.
  CUTLASS_DEVICE DistanceGemmEpilogue(Params const &params_,
                                      SharedStorage &shared_storage_,
                                      Index m_, Index n_)
    : params(params_), shared_storage(shared_storage_), m(m_), n(n_) {}

  /// The memory fence for shared loads.
  CUTLASS_DEVICE void shared_load_fence() { __syncthreads(); }

  /// The memory fence for shared stores.
  CUTLASS_DEVICE void shared_store_fence() { __syncthreads(); }

  /// The params.
  Params const &params;
  /// The shared storage.
  SharedStorage &shared_storage;
  /// The dimensions of the GEMM.
  Index m, n;
}; // end struct DistanceGemmEpilogue


/**
 * @brief Epilogue for Cosine and Expanded L2 distance,
 *  which accesses A^2 and B^2 in global memory,
 *  while passing the global index of C to its EpilogueFunctor
 * @tparam GemmEpilogueTraits_ the traits class to configure this epilogue
 */
template <typename GemmEpilogueTraits_,
          typename BaseClass = DistanceGemmEpilogue<GemmEpilogueTraits_>>
struct ExpandedDistanceGemmEpilogue : public BaseClass {
  using typename BaseClass::Params;
  using typename BaseClass::SharedStorage;
  using typename BaseClass::Index;
  using typename BaseClass::Accumulators;
  using typename BaseClass::Functor;
  using typename BaseClass::Traits;
  using typename BaseClass::GlobalLoadIteratorAA;
  using typename BaseClass::GlobalLoadIteratorBB;
  using typename BaseClass::GlobalStoreIteratorD;
  using typename BaseClass::GlobalTransformerD;
  using typename BaseClass::SharedStoreTransformerD;
  using typename BaseClass::SharedLoadIteratorD;
  using typename BaseClass::SharedStoreIteratorD;
  using typename BaseClass::Iterations;
  using BaseClass::shared_load_fence;
  using BaseClass::shared_store_fence;

  /// Ctor.
  CUTLASS_DEVICE ExpandedDistanceGemmEpilogue(
    Params const &params_, SharedStorage &shared_storage_, Index m_, Index n_)
    : BaseClass(params_, shared_storage_, m_, n_) {}

  /// Execute the epilogue
  template <typename FinalLambda>
  CUTLASS_DEVICE void epilogue(cutlass::Coord<3> const &block,
                               Accumulators &accumulators, FinalLambda fin_op) {
    // The problem size.
    cutlass::Coord<3> const bounds = cutlass::make_Coord(0, this->n, this->m);

    // The functor.
    Functor functor(this->params.functor);

    // The AA fragment
    typename GlobalLoadIteratorAA::Fragment fragment_aa;
    // The BB fragment
    typename GlobalLoadIteratorBB::Fragment fragment_bb;

    // The BB column vector size.
    cutlass::Coord<3> const bb_bounds = cutlass::make_Coord(0, 1, this->m);
    // The BB block size.
    cutlass::Coord<3> const bb_block = cutlass::make_Coord(0, 0, block[2]);

    // The iterator to load the elements of the BB row vector.
    GlobalLoadIteratorBB global_load_iterator_bb(
      this->params.functor.iterator_bb, bb_bounds, bb_block, 0, 0);
    iterator_load(global_load_iterator_bb, fragment_bb);

    // Preserve the base pointer of the output D matrix
    typename GlobalStoreIteratorD::Pointer global_base_ptr =
      this->params.iterator_d.pointer;

    CUTLASS_PRAGMA_UNROLL
    for (int h = 0; h < Iterations::kH; ++h) {
      // Compute pointer and predicate offsets for C and D global iterators.
      int const pointer_offset =
        ((this->params.iterator_d.inc_h * (GlobalStoreIteratorD::Iterations::kH - 1) +
          this->params.iterator_d.inc_advance) *
           Iterations::kW +
         this->params.stride_h) *
        h;
      int const predicate_offset =
        ((this->params.iterator_d.predicate_inc_h *
            (GlobalStoreIteratorD::Iterations::kH - 1) +
          this->params.iterator_d.predicate_inc_advance) *
           Iterations::kW +
         Traits::Delta::kH) *
        h;

      // The transformer for D.
      GlobalTransformerD transformer_d;
      // The iterator to store into the D matrix.
      GlobalStoreIteratorD global_store_iterator(
        this->params.iterator_d, bounds, block, pointer_offset, predicate_offset);

      // The transformer to transform before storing to shared memory.
      SharedStoreTransformerD shared_store_transformer;
      typename SharedStoreTransformerD::OutputFragment
        shared_store_transformed_d;

      // The iterator to store to shared memory.
      SharedStoreIteratorD shared_store_iterator(
        this->params.shared_store_iterator_d,
        this->shared_storage.shared_stream.store);

      // The iterator to load from shared memory. TODO: Use a stream.
      SharedLoadIteratorD shared_load_iterator(
        this->params.shared_load_iterator_d,
        this->shared_storage.shared_stream.load);

      // The AA column vector size.
      cutlass::Coord<3> const aa_bounds = cutlass::make_Coord(0, this->n, 1);
      // The AA block size.
      cutlass::Coord<3> const aa_block = cutlass::make_Coord(0, block[1], 0);

      // Compute pointer and predicate offsets for AA global iterators.
      int const aa_pointer_offset =
        ((this->params.functor.iterator_aa.inc_h *
            (GlobalLoadIteratorAA::Iterations::kH - 1) +
          this->params.functor.iterator_aa.inc_advance) *
           Iterations::kW +
         Traits::Delta::kH) *
        h;
      int const aa_predicate_offset =
        ((this->params.functor.iterator_aa.predicate_inc_h *
            (GlobalLoadIteratorAA::Iterations::kH - 1) +
          this->params.functor.iterator_aa.predicate_inc_advance) *
           Iterations::kW +
         Traits::Delta::kH) *
        h;
      // The iterator to load the elements of the AA column vector.
      GlobalLoadIteratorAA global_load_iterator_aa(
        this->params.functor.iterator_aa, aa_bounds, aa_block, aa_pointer_offset,
        aa_predicate_offset);

      CUTLASS_PRAGMA_UNROLL
      for (int w = 0; w < Iterations::kW; ++w) {
        iterator_load(global_load_iterator_aa, fragment_aa);

        // Make sure we can write to shared memory.
        shared_load_fence();

        // Copy the accumulators to shared memory.
        int const offset =
          (h * Iterations::kW + w) * SharedStoreIteratorD::Fragment::kElements;

        shared_store_transformer.transform(accumulators, offset,
                                           shared_store_transformed_d);
        shared_iterator_store(shared_store_iterator,
                              shared_store_transformed_d);

        // Make sure the data is in shared memory.
        shared_store_fence();

        // Copy the accumulators back to registers from shared memory.
        typename SharedLoadIteratorD::Fragment fetched_d;
        shared_iterator_load(shared_load_iterator, fetched_d);

        // Do the math.
        typename GlobalTransformerD::InputFragment fragment_d;

        // extract the global pointer index for each fragment element
        int index[GlobalStoreIteratorD::Fragment::kElements];
        extract_index_from_iterator(global_store_iterator, global_base_ptr,
                                    index);

        functor.evaluate(fetched_d, fragment_d, index, fragment_aa,
                         fragment_bb, fin_op);

        // Transform D fragment.
        typename GlobalTransformerD::OutputFragment transformed_d;
        transformer_d.transform(fragment_d, transformed_d);

        // Copy the results to global memory.
        iterator_store(global_store_iterator, transformed_d);
      }
    }
  }
}; // end struct ExpandedDistanceGemmEpilogue


/**
 * @brief Epilogue for L1 and Unexpanded L2 distance,
 *  which passes the global index of C to its EpilogueFunctor
 * @tparam GemmEpilogueTraits_ the traits class to configure this epilogue
 */
template <typename GemmEpilogueTraits_,
          typename BaseClass = DistanceGemmEpilogue<GemmEpilogueTraits_>>
struct UnexpandedDistanceGemmEpilogue : public BaseClass {
  using typename BaseClass::Params;
  using typename BaseClass::SharedStorage;
  using typename BaseClass::Index;
  using typename BaseClass::Accumulators;
  using typename BaseClass::Functor;
  using typename BaseClass::Traits;
  using typename BaseClass::GlobalLoadIteratorAA;
  using typename BaseClass::GlobalLoadIteratorBB;
  using typename BaseClass::GlobalStoreIteratorD;
  using typename BaseClass::GlobalTransformerD;
  using typename BaseClass::SharedStoreTransformerD;
  using typename BaseClass::SharedLoadIteratorD;
  using typename BaseClass::SharedStoreIteratorD;
  using typename BaseClass::Iterations;
  using BaseClass::shared_load_fence;
  using BaseClass::shared_store_fence;

  /// Ctor.
  CUTLASS_DEVICE UnexpandedDistanceGemmEpilogue(
    Params const &params_, SharedStorage &shared_storage_, Index m_, Index n_)
    : BaseClass(params_, shared_storage_, m_, n_) {}

  /// Execute the epilogue
  template <typename FinalLambda>
  CUTLASS_DEVICE void epilogue(cutlass::Coord<3> const &block,
                               Accumulators &accumulators, FinalLambda fin_op) {
    // The problem size.
    cutlass::Coord<3> const bounds = cutlass::make_Coord(0, this->n, this->m);

    // The functor.
    Functor functor(this->params.functor);

    // Preserve the base pointer of the output D matrix
    typename GlobalStoreIteratorD::Pointer global_base_ptr =
      this->params.iterator_d.pointer;

    CUTLASS_PRAGMA_UNROLL
    for (int h = 0; h < Iterations::kH; ++h) {
      // Compute pointer and predicate offsets for C and D global iterators.
      int const pointer_offset =
        ((this->params.iterator_d.inc_h * (GlobalStoreIteratorD::Iterations::kH - 1) +
          this->params.iterator_d.inc_advance) *
           Iterations::kW +
         this->params.stride_h) *
        h;
      int const predicate_offset =
        ((this->params.iterator_d.predicate_inc_h *
            (GlobalStoreIteratorD::Iterations::kH - 1) +
          this->params.iterator_d.predicate_inc_advance) *
           Iterations::kW +
         Traits::Delta::kH) *
        h;

      // The transformer for D.
      GlobalTransformerD transformer_d;
      // The iterator to store into the D matrix.
      GlobalStoreIteratorD global_store_iterator(
        this->params.iterator_d, bounds, block, pointer_offset, predicate_offset);

      // The transformer to transform before storing to shared memory.
      SharedStoreTransformerD shared_store_transformer;
      typename SharedStoreTransformerD::OutputFragment
        shared_store_transformed_d;

      // The iterator to store to shared memory.
      SharedStoreIteratorD shared_store_iterator(
        this->params.shared_store_iterator_d,
        this->shared_storage.shared_stream.store);

      // The iterator to load from shared memory. TODO: Use a stream.
      SharedLoadIteratorD shared_load_iterator(
        this->params.shared_load_iterator_d,
        this->shared_storage.shared_stream.load);

      CUTLASS_PRAGMA_UNROLL
      for (int w = 0; w < Iterations::kW; ++w) {
        // Make sure we can write to shared memory.
        shared_load_fence();

        // Copy the accumulators to shared memory.
        int const offset =
          (h * Iterations::kW + w) * SharedStoreIteratorD::Fragment::kElements;

        shared_store_transformer.transform(accumulators, offset,
                                           shared_store_transformed_d);
        shared_iterator_store(shared_store_iterator,
                              shared_store_transformed_d);

        // Make sure the data is in shared memory.
        shared_store_fence();

        // Copy the accumulators back to registers from shared memory.
        typename SharedLoadIteratorD::Fragment fetched_d;
        shared_iterator_load(shared_load_iterator, fetched_d);

        // Do the math.
        typename GlobalTransformerD::InputFragment fragment_d;

        // extract the global pointer index for each fragment element
        int index[GlobalStoreIteratorD::Fragment::kElements];
        extract_index_from_iterator(global_store_iterator, global_base_ptr,
                                    index);

        functor.evaluate(fetched_d, fragment_d, index, fin_op);

        // Transform D fragment.
        typename GlobalTransformerD::OutputFragment transformed_d;
        transformer_d.transform(fragment_d, transformed_d);

        // Copy the results to global memory.
        iterator_store(global_store_iterator, transformed_d);
      }
    }
  }
}; // end struct UnexpandedDistanceGemmEpilogue


} // end namespace Distance
} // end namespace MLCommon

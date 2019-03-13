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

#include <cutlass/gemm/gemm_epilogue_traits.h>
#include <cutlass/gemm/gemm_global_tile.h>

namespace MLCommon {
namespace Distance {

template <
    typename GemmConfig_,
    typename EpilogueFunctor_,
    typename Index_ = int,
    typename BaseClass =
      cutlass::gemm::GemmEpilogueTraitsHelper<
          GemmConfig_, EpilogueFunctor_, Index_>>
struct BoolEpilogueTraitsHelper : public BaseClass {
  typedef typename BaseClass::Scalar Scalar;
  typedef typename BaseClass::OutputTile OutputTile;
  typedef typename BaseClass::Iterations Iterations;
  typedef typename BaseClass::Functor Functor;

  typedef typename BaseClass::SharedStoreTileTraits SharedStoreTileTraits;
  typedef typename BaseClass::SharedStoreIteratorD SharedStoreIteratorD;
  typedef typename BaseClass::SharedStoreTransformerD SharedStoreTransformerD;
  typedef typename BaseClass::SharedLoadTileTraits SharedLoadTileTraits;
  typedef typename BaseClass::SharedLoadIteratorD SharedLoadIteratorD;

  typedef typename BaseClass::GlobalLoadTileTraits GlobalLoadTileTraits;
  typedef typename BaseClass::GlobalLoadIteratorC GlobalLoadIteratorC;
  typedef typename BaseClass::GlobalTransformerC GlobalTransformerC;

  /// The traits class to build the iterator to store data to global memory for D^N.
  typedef cutlass::gemm::GemmGlobalTileCdTraits<
      bool,
      // The tile has size (N / Iterations)xM in GEMM's terminology.
      cutlass::Shape<1,
            GemmConfig_::OutputTile::kH / cutlass::ShapeCount<Iterations>::kCount,
            GemmConfig_::OutputTile::kW>,
      // The threads are distributed as warps x 32 (the traits may reorganize).
      cutlass::Shape<1, cutlass::ShapeCount<typename GemmConfig_::Warps>::kCount, GemmConfig_::kWarpSize>,
      // How many elements do we jump over at each iteration?
      Iterations::kW,
      // The number of scalars per LDG (LDG.32 or LDG.128, etc).
      GemmConfig_::kScalarsPerStgD>
      GlobalStoreTileTraits;

  /// The iterator to store D.
  typedef cutlass::gemm::GemmGlobalIteratorCd<GlobalStoreTileTraits, Index_> GlobalStoreIteratorD;
  /// The transformer for D.
  typedef cutlass::Convert<
      cutlass::Fragment<typename GemmConfig_::ScalarD, GlobalStoreIteratorD::Fragment::kElements>,
      typename GlobalStoreIteratorD::Fragment
  > GlobalTransformerD;
      // cutlass::ShapeCount<GlobalStoreIteratorD::Iterations>::kCount * GlobalStoreIteratorD::Tile::kC> lobalTransformerD;
};

}  // namespace Distance
}  // namespace MLCommon

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

#include <cutlass/coord.h>
#include <cutlass/gemm/gemm_operand.h>
#include <cutlass/gemm/gemm_global_tile.h>
#include <cutlass/matrix_traits.h>
#include <cutlass/shape.h>

namespace MLCommon {
namespace Distance {

template <typename Scalar_, typename Tile_, typename Threads_, int kStrideH_, int kAccessSize_>
struct DistanceGlobalTileAATraits
    : public cutlass::gemm::GemmGlobalTileTraits<
      cutlass::GemmOperand::kA,
      cutlass::MatrixLayout::kRowMajor,
      Scalar_,
      Tile_,
      Threads_,
      kAccessSize_> {
  /// The base class.
  typedef cutlass::gemm::GemmGlobalTileTraits<cutlass::GemmOperand::kA,
                               cutlass::MatrixLayout::kRowMajor,
                               Scalar_,
                               Tile_,
                               Threads_,
                               kAccessSize_>
      Base;

  /// The stride in the H dimension.
  static int const kStrideH = kStrideH_;
  /// Override the strides in each dimension between different loads/stores.
  typedef cutlass::Shape<0, 0, 0, Base::Delta::kC> Delta;

  /// Override the number of iterations needed to load/store the tile.
  typedef cutlass::Shape<1, Tile_::kH / Threads_::kH, 1, Tile_::kC / kAccessSize_>
      Iterations;

  typedef typename Base::Threads Threads;

  typedef typename Base::ThreadsDelta ThreadsDelta;

  typedef typename Base::ImmediateOffsetStrides ImmediateOffsetStrides;

  /// Computes the thread offset in (H, W) based on thread ID
  struct ThreadOffset {
    CUTLASS_HOST_DEVICE
    cutlass::Coord<4> operator()() const {
      int thread_offset_h = threadIdx.x / Threads::kW * kStrideH * Iterations::kH;
      int thread_offset_w = 0;

      return cutlass::make_Coord(0, thread_offset_h, thread_offset_w, 0);
    }
  };
};

template <typename Scalar_, typename Tile_, typename Threads_, int kAccessSize_>
struct DistanceGlobalTileBBTraits
    : public cutlass::gemm::GemmGlobalTileTraits<
      cutlass::GemmOperand::kB,
      cutlass::MatrixLayout::kColumnMajor,
      Scalar_,
      Tile_,
      Threads_,
      kAccessSize_> {
  /// The base class.
  typedef cutlass::gemm::GemmGlobalTileTraits<cutlass::GemmOperand::kB,
                               cutlass::MatrixLayout::kColumnMajor,
                               Scalar_,
                               Tile_,
                               Threads_,
                               kAccessSize_>
      Base;

  /// The stride in the H dimension.
  static int const kStrideH = 0;
  /// Override the strides in each dimension between different loads/stores.
  typedef cutlass::Shape<0, 0, Base::Delta::kW, Base::Delta::kC> Delta;

  /// Override the number of iterations needed to load/store the tile.
  typedef cutlass::Shape<1, 1, Tile_::kW / Threads_::kW, Tile_::kC / kAccessSize_>
      Iterations;

  typedef typename Base::Threads Threads;

  typedef typename Base::ThreadsDelta ThreadsDelta;

  typedef typename Base::ImmediateOffsetStrides ImmediateOffsetStrides;

  /// Computes the thread offset in (H, W) based on thread ID
  struct ThreadOffset {
    CUTLASS_HOST_DEVICE
    cutlass::Coord<4> operator()() const {
      int thread_offset_h = 0;
      int thread_offset_w = threadIdx.x % Threads::kW * ThreadsDelta::kW;

      return cutlass::make_Coord(0, thread_offset_h, thread_offset_w, 0);
    }
  };
};

} // end namespace Distance
} // end namespace MLCommon

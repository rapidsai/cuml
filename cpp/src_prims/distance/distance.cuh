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

#include <cuda_runtime_api.h>
#include <cutlass/shape.h>
#include <raft/linalg/distance_type.h>
#include <cuml/common/device_buffer.hpp>
#include <raft/cuda_utils.cuh>
#include "cosine.cuh"
#include "euclidean.cuh"
#include "l1.cuh"

namespace MLCommon {
namespace Distance {

typedef cutlass::Shape<8, 128, 128> OutputTile_8x128x128;

namespace {
template <raft::distance::DistanceType distanceType, typename InType,
          typename AccType, typename OutType, typename OutputTile_,
          typename FinalLambda, typename Index_>
struct DistanceImpl {
  void run(const InType *x, const InType *y, OutType *dist, Index_ m, Index_ n,
           Index_ k, void *workspace, size_t worksize, FinalLambda fin_op,
           cudaStream_t stream, bool isRowMajor) {}
};

template <typename InType, typename AccType, typename OutType,
          typename OutputTile_, typename FinalLambda, typename Index_>
struct DistanceImpl<raft::distance::DistanceType::L2Expanded, InType, AccType,
                    OutType, OutputTile_, FinalLambda, Index_> {
  void run(const InType *x, const InType *y, OutType *dist, Index_ m, Index_ n,
           Index_ k, void *workspace, size_t worksize, FinalLambda fin_op,
           cudaStream_t stream, bool isRowMajor) {
    euclideanAlgo1<InType, AccType, OutType, OutputTile_, FinalLambda, Index_>(
      m, n, k, x, y, dist, false, (AccType *)workspace, worksize, fin_op,
      stream, isRowMajor);
  }
};

template <typename InType, typename AccType, typename OutType,
          typename OutputTile_, typename FinalLambda, typename Index_>
struct DistanceImpl<raft::distance::DistanceType::L2SqrtExpanded, InType,
                    AccType, OutType, OutputTile_, FinalLambda, Index_> {
  void run(const InType *x, const InType *y, OutType *dist, Index_ m, Index_ n,
           Index_ k, void *workspace, size_t worksize, FinalLambda fin_op,
           cudaStream_t stream, bool isRowMajor) {
    euclideanAlgo1<InType, AccType, OutType, OutputTile_, FinalLambda, Index_>(
      m, n, k, x, y, dist, true, (AccType *)workspace, worksize, fin_op, stream,
      isRowMajor);
  }
};

template <typename InType, typename AccType, typename OutType,
          typename OutputTile_, typename FinalLambda, typename Index_>
struct DistanceImpl<raft::distance::DistanceType::CosineExpanded, InType,
                    AccType, OutType, OutputTile_, FinalLambda, Index_> {
  void run(const InType *x, const InType *y, OutType *dist, Index_ m, Index_ n,
           Index_ k, void *workspace, size_t worksize, FinalLambda fin_op,
           cudaStream_t stream, bool isRowMajor) {
    cosineAlgo1<InType, AccType, OutType, OutputTile_, FinalLambda, Index_>(
      m, n, k, x, y, dist, (AccType *)workspace, worksize, fin_op, stream,
      isRowMajor);
  }
};

template <typename InType, typename AccType, typename OutType,
          typename OutputTile_, typename FinalLambda, typename Index_>
struct DistanceImpl<raft::distance::DistanceType::L2Unexpanded, InType, AccType,
                    OutType, OutputTile_, FinalLambda, Index_> {
  void run(const InType *x, const InType *y, OutType *dist, Index_ m, Index_ n,
           Index_ k, void *workspace, size_t worksize, FinalLambda fin_op,
           cudaStream_t stream, bool isRowMajor) {
    euclideanAlgo2<InType, AccType, OutType, OutputTile_, FinalLambda, Index_>(
      m, n, k, x, y, dist, false, fin_op, stream, isRowMajor);
  }
};

template <typename InType, typename AccType, typename OutType,
          typename OutputTile_, typename FinalLambda, typename Index_>
struct DistanceImpl<raft::distance::DistanceType::L2SqrtUnexpanded, InType,
                    AccType, OutType, OutputTile_, FinalLambda, Index_> {
  void run(const InType *x, const InType *y, OutType *dist, Index_ m, Index_ n,
           Index_ k, void *workspace, size_t worksize, FinalLambda fin_op,
           cudaStream_t stream, bool isRowMajor) {
    euclideanAlgo2<InType, AccType, OutType, OutputTile_, FinalLambda, Index_>(
      m, n, k, x, y, dist, true, fin_op, stream, isRowMajor);
  }
};

template <typename InType, typename AccType, typename OutType,
          typename OutputTile_, typename FinalLambda, typename Index_>
struct DistanceImpl<raft::distance::DistanceType::L1, InType, AccType, OutType,
                    OutputTile_, FinalLambda, Index_> {
  void run(const InType *x, const InType *y, OutType *dist, Index_ m, Index_ n,
           Index_ k, void *workspace, size_t worksize, FinalLambda fin_op,
           cudaStream_t stream, bool isRowMajor) {
    l1Impl<InType, AccType, OutType, OutputTile_, FinalLambda, Index_>(
      m, n, k, x, y, dist, fin_op, stream, isRowMajor);
  }
};

}  // anonymous namespace

/**
 * @brief Return the exact workspace size to compute the distance
 * @tparam DistanceType which distance to evaluate
 * @tparam InType input argument type
 * @tparam AccType accumulation type
 * @tparam OutType output type
 * @tparam Index_ Index type
 * @param x first set of points
 * @param y second set of points
 * @param m number of points in x
 * @param n number of points in y
 * @param k dimensionality
 *
 * @note If the specifed distanceType doesn't need the workspace at all, it
 * returns 0.
 */
template <raft::distance::DistanceType distanceType, typename InType,
          typename AccType, typename OutType, typename Index_ = int>
size_t getWorkspaceSize(const InType *x, const InType *y, Index_ m, Index_ n,
                        Index_ k) {
  size_t worksize = 0;
  constexpr bool is_allocated =
    distanceType <= raft::distance::DistanceType::CosineExpanded;
  if (is_allocated) {
    worksize += m * sizeof(AccType);
    if (x != y) worksize += n * sizeof(AccType);
  }
  return worksize;
}

#if CUDART_VERSION >= 10010
// With optimization enabled, CUDA 10.1 generates segfaults for distance
// prims, so disable optimization until another workaround is found
// #pragma GCC push_options
#pragma GCC optimize("O0")
#endif

/**
 * @brief Evaluate pairwise distances with the user epilogue lamba allowed
 * @tparam DistanceType which distance to evaluate
 * @tparam InType input argument type
 * @tparam AccType accumulation type
 * @tparam OutType output type
 * @tparam FinalLambda user-defined epilogue lamba
 * @tparam Index_ Index type
 * @param x first set of points
 * @param y second set of points
 * @param dist output distance matrix
 * @param m number of points in x
 * @param n number of points in y
 * @param k dimensionality
 * @param workspace temporary workspace needed for computations
 * @param worksize number of bytes of the workspace
 * @param fin_op the final gemm epilogue lambda
 * @param stream cuda stream
 * @param isRowMajor whether the matrices are row-major or col-major
 *
 * @note fin_op: This is a device lambda which is supposed to operate upon the
 * input which is AccType and returns the output in OutType. It's signature is
 * as follows:  <pre>OutType fin_op(AccType in, int g_idx);</pre>. If one needs
 * any other parameters, feel free to pass them via closure.
 */
template <raft::distance::DistanceType distanceType, typename InType,
          typename AccType, typename OutType, typename OutputTile_,
          typename FinalLambda, typename Index_ = int>
void distance(const InType *x, const InType *y, OutType *dist, Index_ m,
              Index_ n, Index_ k, void *workspace, size_t worksize,
              FinalLambda fin_op, cudaStream_t stream, bool isRowMajor = true) {
  DistanceImpl<distanceType, InType, AccType, OutType, OutputTile_, FinalLambda,
               Index_>
    distImpl;
  distImpl.run(x, y, dist, m, n, k, workspace, worksize, fin_op, stream,
               isRowMajor);
  CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief Evaluate pairwise distances for the simple use case
 * @tparam DistanceType which distance to evaluate
 * @tparam InType input argument type
 * @tparam AccType accumulation type
 * @tparam OutType output type
 * @tparam Index_ Index type
 * @param x first set of points
 * @param y second set of points
 * @param dist output distance matrix
 * @param m number of points in x
 * @param n number of points in y
 * @param k dimensionality
 * @param workspace temporary workspace needed for computations
 * @param worksize number of bytes of the workspace
 * @param stream cuda stream
 * @param isRowMajor whether the matrices are row-major or col-major
 *
 * @note if workspace is passed as nullptr, this will return in
 *  worksize, the number of bytes of workspace required
 */
template <raft::distance::DistanceType distanceType, typename InType,
          typename AccType, typename OutType, typename OutputTile_,
          typename Index_ = int>
void distance(const InType *x, const InType *y, OutType *dist, Index_ m,
              Index_ n, Index_ k, void *workspace, size_t worksize,
              cudaStream_t stream, bool isRowMajor = true) {
  auto default_fin_op = [] __device__(AccType d_val, Index_ g_d_idx) {
    return d_val;
  };
  distance<distanceType, InType, AccType, OutType, OutputTile_,
           decltype(default_fin_op), Index_>(x, y, dist, m, n, k, workspace,
                                             worksize, default_fin_op, stream,
                                             isRowMajor);
  CUDA_CHECK(cudaPeekAtLastError());
}

#if CUDART_VERSION >= 10010
// Undo special optimization options set earlier
#pragma GCC reset_options
#endif

/**
 * @defgroup pairwise_distance pairwise distance prims
 * @{
 * @brief Convenience wrapper around 'distance' prim to convert runtime metric
 * into compile time for the purpose of dispatch
 * @tparam Type input/accumulation/output data-type
 * @tparam Index_ indexing type
 * @param x first set of points
 * @param y second set of points
 * @param dist output distance matrix
 * @param m number of points in x
 * @param n number of points in y
 * @param k dimensionality
 * @param workspace temporary workspace buffer which can get resized as per the
 * needed workspace size
 * @param metric distance metric
 * @param stream cuda stream
 * @param isRowMajor whether the matrices are row-major or col-major
 */
template <typename Type, typename Index_, raft::distance::DistanceType DistType>
void pairwise_distance_impl(const Type *x, const Type *y, Type *dist, Index_ m,
                            Index_ n, Index_ k, device_buffer<char> &workspace,
                            cudaStream_t stream, bool isRowMajor) {
  auto worksize =
    getWorkspaceSize<DistType, Type, Type, Type, Index_>(x, y, m, n, k);
  workspace.resize(worksize, stream);
  distance<DistType, Type, Type, Type, OutputTile_8x128x128, Index_>(
    x, y, dist, m, n, k, workspace.data(), worksize, stream, isRowMajor);
}

template <typename Type, typename Index_ = int>
void pairwise_distance(const Type *x, const Type *y, Type *dist, Index_ m,
                       Index_ n, Index_ k, device_buffer<char> &workspace,
                       raft::distance::DistanceType metric, cudaStream_t stream,
                       bool isRowMajor = true) {
  switch (metric) {
    case raft::distance::DistanceType::L2Expanded:
      pairwise_distance_impl<Type, Index_,
                             raft::distance::DistanceType::L2Expanded>(
        x, y, dist, m, n, k, workspace, stream, isRowMajor);
      break;
    case raft::distance::DistanceType::L2SqrtExpanded:
      pairwise_distance_impl<Type, Index_,
                             raft::distance::DistanceType::L2SqrtExpanded>(
        x, y, dist, m, n, k, workspace, stream, isRowMajor);
      break;
    case raft::distance::DistanceType::CosineExpanded:
      pairwise_distance_impl<Type, Index_,
                             raft::distance::DistanceType::CosineExpanded>(
        x, y, dist, m, n, k, workspace, stream, isRowMajor);
      break;
    case raft::distance::DistanceType::L1:
      pairwise_distance_impl<Type, Index_, raft::distance::DistanceType::L1>(
        x, y, dist, m, n, k, workspace, stream, isRowMajor);
      break;
    case raft::distance::DistanceType::L2Unexpanded:
      pairwise_distance_impl<Type, Index_,
                             raft::distance::DistanceType::L2Unexpanded>(
        x, y, dist, m, n, k, workspace, stream, isRowMajor);
      break;
    case raft::distance::DistanceType::L2SqrtUnexpanded:
      pairwise_distance_impl<Type, Index_,
                             raft::distance::DistanceType::L2SqrtUnexpanded>(
        x, y, dist, m, n, k, workspace, stream, isRowMajor);
      break;
    default:
      THROW("Unknown or unsupported distance metric '%d'!", (int)metric);
  };
}
/** @} */

};  // end namespace Distance
};  // end namespace MLCommon

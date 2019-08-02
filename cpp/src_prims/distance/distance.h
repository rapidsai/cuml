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

#include <cutlass/shape.h>
#include "common/device_buffer.hpp"
#include "cuda_utils.h"
#include "distance/cosine.h"
#include "distance/euclidean.h"
#include "distance/l1.h"

namespace MLCommon {
namespace Distance {

typedef cutlass::Shape<8, 128, 128> OutputTile_8x128x128;

/** enum to tell how to compute euclidean distance */
enum DistanceType {
  /** evaluate as dist_ij = sum(x_ik^2) + sum(y_ij)^2 - 2*sum(x_ik * y_jk) */
  EucExpandedL2 = 0,
  /** same as above, but inside the epilogue, perform square root operation */
  EucExpandedL2Sqrt,
  /** cosine distance */
  EucExpandedCosine,
  /** L1 distance */
  EucUnexpandedL1,
  /** evaluate as dist_ij += (x_ik - y-jk)^2 */
  EucUnexpandedL2,
  /** same as above, but inside the epilogue, perform square root operation */
  EucUnexpandedL2Sqrt,
};

namespace {
template <DistanceType distanceType, typename InType, typename AccType,
          typename OutType, typename OutputTile_, typename FinalLambda,
          typename Index_>
struct DistanceImpl {
  void run(const InType *x, const InType *y, OutType *dist, Index_ m, Index_ n,
           Index_ k, void *workspace, size_t worksize, FinalLambda fin_op,
           cudaStream_t stream, bool isRowMajor) {}
};

template <typename InType, typename AccType, typename OutType,
          typename OutputTile_, typename FinalLambda, typename Index_>
struct DistanceImpl<EucExpandedL2, InType, AccType, OutType, OutputTile_,
                    FinalLambda, Index_> {
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
struct DistanceImpl<EucExpandedL2Sqrt, InType, AccType, OutType, OutputTile_,
                    FinalLambda, Index_> {
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
struct DistanceImpl<EucExpandedCosine, InType, AccType, OutType, OutputTile_,
                    FinalLambda, Index_> {
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
struct DistanceImpl<EucUnexpandedL2, InType, AccType, OutType, OutputTile_,
                    FinalLambda, Index_> {
  void run(const InType *x, const InType *y, OutType *dist, Index_ m, Index_ n,
           Index_ k, void *workspace, size_t worksize, FinalLambda fin_op,
           cudaStream_t stream, bool isRowMajor) {
    euclideanAlgo2<InType, AccType, OutType, OutputTile_, FinalLambda, Index_>(
      m, n, k, x, y, dist, false, fin_op, stream, isRowMajor);
  }
};

template <typename InType, typename AccType, typename OutType,
          typename OutputTile_, typename FinalLambda, typename Index_>
struct DistanceImpl<EucUnexpandedL2Sqrt, InType, AccType, OutType, OutputTile_,
                    FinalLambda, Index_> {
  void run(const InType *x, const InType *y, OutType *dist, Index_ m, Index_ n,
           Index_ k, void *workspace, size_t worksize, FinalLambda fin_op,
           cudaStream_t stream, bool isRowMajor) {
    euclideanAlgo2<InType, AccType, OutType, OutputTile_, FinalLambda, Index_>(
      m, n, k, x, y, dist, true, fin_op, stream, isRowMajor);
  }
};

template <typename InType, typename AccType, typename OutType,
          typename OutputTile_, typename FinalLambda, typename Index_>
struct DistanceImpl<EucUnexpandedL1, InType, AccType, OutType, OutputTile_,
                    FinalLambda, Index_> {
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
template <DistanceType distanceType, typename InType, typename AccType,
          typename OutType, typename Index_ = int>
size_t getWorkspaceSize(const InType *x, const InType *y, Index_ m, Index_ n,
                        Index_ k) {
  size_t worksize = 0;
  constexpr bool is_allocated = distanceType <= EucExpandedCosine;
  if (is_allocated) {
    worksize += m * sizeof(AccType);
    if (x != y) worksize += n * sizeof(AccType);
  }
  return worksize;
}

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
template <DistanceType distanceType, typename InType, typename AccType,
          typename OutType, typename OutputTile_, typename FinalLambda,
          typename Index_ = int>
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
template <DistanceType distanceType, typename InType, typename AccType,
          typename OutType, typename OutputTile_, typename Index_ = int>
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

/**
 * @defgroup PairwiseDistance
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
 * @needed workspace size
 * @param metric distance metric
 * @param stream cuda stream
 * @param isRowMajor whether the matrices are row-major or col-major
 */
template <typename Type, typename Index_, DistanceType DistType>
void pairwiseDistanceImpl(const Type *x, const Type *y, Type *dist, Index_ m,
                          Index_ n, Index_ k, device_buffer<char> &workspace,
                          cudaStream_t stream, bool isRowMajor) {
  auto worksize =
    getWorkspaceSize<DistType, Type, Type, Type, Index_>(x, y, m, n, k);
  workspace.resize(worksize, stream);
  distance<DistType, Type, Type, Type, OutputTile_8x128x128, Index_>(
    x, y, dist, m, n, k, workspace.data(), worksize, stream, isRowMajor);
}

template <typename Type, typename Index_ = int>
void pairwiseDistance(const Type *x, const Type *y, Type *dist, Index_ m,
                      Index_ n, Index_ k, device_buffer<char> &workspace,
                      DistanceType metric, cudaStream_t stream,
                      bool isRowMajor = true) {
  switch (metric) {
    case DistanceType::EucExpandedL2:
      pairwiseDistanceImpl<Type, Index_, DistanceType::EucExpandedL2>(
        x, y, dist, m, n, k, workspace, stream, isRowMajor);
      break;
    case DistanceType::EucExpandedL2Sqrt:
      pairwiseDistanceImpl<Type, Index_, DistanceType::EucExpandedL2Sqrt>(
        x, y, dist, m, n, k, workspace, stream, isRowMajor);
      break;
    case DistanceType::EucExpandedCosine:
      pairwiseDistanceImpl<Type, Index_, DistanceType::EucExpandedCosine>(
        x, y, dist, m, n, k, workspace, stream, isRowMajor);
      break;
    case DistanceType::EucUnexpandedL1:
      pairwiseDistanceImpl<Type, Index_, DistanceType::EucUnexpandedL1>(
        x, y, dist, m, n, k, workspace, stream, isRowMajor);
      break;
    case DistanceType::EucUnexpandedL2:
      pairwiseDistanceImpl<Type, Index_, DistanceType::EucUnexpandedL2>(
        x, y, dist, m, n, k, workspace, stream, isRowMajor);
      break;
    case DistanceType::EucUnexpandedL2Sqrt:
      pairwiseDistanceImpl<Type, Index_, DistanceType::EucUnexpandedL2Sqrt>(
        x, y, dist, m, n, k, workspace, stream, isRowMajor);
      break;
    default:
      THROW("Unknown distance metric '%d'!", metric);
  };
}
/** @} */

/**
 * @brief Constructs an epsilon neighborhood adjacency matrix by
 * filtering the final distance by some epsilon.
 * @tparam distanceType: distance metric to compute between a and b matrices
 * @tparam T: the type of input matrices a and b
 * @tparam Lambda Lambda function
 * @tparam Index_ Index type
 * @tparam OutputTile_ output tile size per thread
 * @param a: row-major input matrix a
 * @param b: row-major input matrix b
 * @param adj: a boolean output adjacency matrix
 * @param m: number of points in a
 * @param n: number of points in b
 * @param k: dimensionality
 * @param eps: the epsilon value to use as a filter for neighborhood construction.
 *             it is important to note that if the distance type returns a squared
 *             variant for efficiency, the epsilon will need to be squared as well.
 * @param workspace: temporary workspace needed for computations
 * @param worksize: number of bytes of the workspace
 * @param stream cuda stream
 * @param fused_op: optional functor taking the output index into c
 *                  and a boolean denoting whether or not the inputs are part of
 *                  the epsilon neighborhood.
 */
template <DistanceType distanceType, typename T, typename Lambda,
          typename Index_ = int, typename OutputTile_ = OutputTile_8x128x128>
size_t epsilon_neighborhood(const T *a, const T *b, bool *adj, Index_ m,
                            Index_ n, Index_ k, T eps, void *workspace,
                            size_t worksize, cudaStream_t stream,
                            Lambda fused_op) {
  auto epsilon_op = [n, eps, fused_op] __device__(T val, Index_ global_c_idx) {
    bool acc = val <= eps;
    fused_op(global_c_idx, acc);
    return acc;
  };

  distance<distanceType, T, T, bool, OutputTile_, decltype(epsilon_op), Index_>(
    a, b, adj, m, n, k, (void *)workspace, worksize, epsilon_op, stream);

  return worksize;
}

/**
 * @brief Constructs an epsilon neighborhood adjacency matrix by
 * filtering the final distance by some epsilon.
 * @tparam distanceType: distance metric to compute between a and b matrices
 * @tparam T: the type of input matrices a and b
 * @tparam Index_ Index type
 * @tparam OutputTile_ output tile size per thread
 * @param a: row-major input matrix a
 * @param b: row-major input matrix b
 * @param adj: a boolean output adjacency matrix
 * @param m: number of points in a
 * @param n: number of points in b
 * @param k: dimensionality
 * @param eps: the epsilon value to use as a filter for neighborhood construction.
 *             it is important to note that if the distance type returns a squared
 *             variant for efficiency, the epsilon will need to be squared as well.
 * @param workspace: temporary workspace needed for computations
 * @param worksize: number of bytes of the workspace
 * @param stream cuda stream
 */
template <DistanceType distanceType, typename T, typename Index_ = int,
          typename OutputTile_ = OutputTile_8x128x128>
size_t epsilon_neighborhood(const T *a, const T *b, bool *adj, Index_ m,
                            Index_ n, Index_ k, T eps, void *workspace,
                            size_t worksize, cudaStream_t stream) {
  auto lambda = [] __device__(Index_ c_idx, bool acc) {};
  return epsilon_neighborhood<distanceType, T, decltype(lambda), Index_,
                              OutputTile_>(a, b, adj, m, n, k, eps, workspace,
                                           worksize, stream, lambda);
}

};  // end namespace Distance
};  // end namespace MLCommon

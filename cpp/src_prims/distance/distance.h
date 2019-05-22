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
#include "cuda_utils.h"
#include "distance/cosine.h"
#include "distance/euclidean.h"
#include "distance/l1.h"

namespace MLCommon {
namespace Distance {

typedef cutlass::Shape<8, 128, 128> OutputTile_t;

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
template <DistanceType distanceType, typename InType, typename AccType, typename OutType,
          typename OutputTile_, typename FinalLambda>
struct DistanceImpl {
  void run(InType *x, InType *y, OutType *dist, int m, int n, int k,
                void *workspace, size_t worksize,
                FinalLambda fin_op, cudaStream_t stream) {}
};

template <typename InType, typename AccType, typename OutType,
          typename OutputTile_, typename FinalLambda>
struct DistanceImpl<EucExpandedL2, InType, AccType, OutType, OutputTile_, FinalLambda> {
  void run(InType *x, InType *y, OutType *dist, int m, int n, int k,
                void *workspace, size_t worksize,
                FinalLambda fin_op, cudaStream_t stream) {
    euclideanAlgo1<InType, AccType, OutType, OutputTile_>(
      m, n, k, x, y, dist, false, (AccType *)workspace, worksize, fin_op,
      stream);
  }
};

template <typename InType, typename AccType, typename OutType,
          typename OutputTile_, typename FinalLambda>
struct DistanceImpl<EucExpandedL2Sqrt, InType, AccType, OutType, OutputTile_, FinalLambda> {
  void run(InType *x, InType *y, OutType *dist, int m, int n, int k,
                void *workspace, size_t worksize,
                FinalLambda fin_op, cudaStream_t stream) {
    euclideanAlgo1<InType, AccType, OutType, OutputTile_>(
      m, n, k, x, y, dist, true, (AccType *)workspace, worksize, fin_op,
      stream);
  }
};

template <typename InType, typename AccType, typename OutType,
          typename OutputTile_, typename FinalLambda>
struct DistanceImpl<EucExpandedCosine, InType, AccType, OutType, OutputTile_, FinalLambda> {
  void run(InType *x, InType *y, OutType *dist, int m, int n, int k,
                void *workspace, size_t worksize,
                FinalLambda fin_op, cudaStream_t stream) {
    cosineAlgo1<InType, AccType, OutType, OutputTile_>(
      m, n, k, x, y, dist, (AccType *)workspace, worksize, fin_op, stream);
  }
};

template <typename InType, typename AccType, typename OutType,
          typename OutputTile_, typename FinalLambda>
struct DistanceImpl<EucUnexpandedL2, InType, AccType, OutType, OutputTile_, FinalLambda> {
  void run(InType *x, InType *y, OutType *dist, int m, int n, int k,
                void *workspace, size_t worksize,
                FinalLambda fin_op, cudaStream_t stream) {
    euclideanAlgo2<InType, AccType, OutType, OutputTile_>(
      m, n, k, x, y, dist, false, fin_op, stream);
  }
};

template <typename InType, typename AccType, typename OutType,
          typename OutputTile_, typename FinalLambda>
struct DistanceImpl<EucUnexpandedL2Sqrt, InType, AccType, OutType, OutputTile_, FinalLambda> {
  void run(InType *x, InType *y, OutType *dist, int m, int n, int k,
                void *workspace, size_t worksize,
                FinalLambda fin_op, cudaStream_t stream) {
    euclideanAlgo2<InType, AccType, OutType, OutputTile_>(
      m, n, k, x, y, dist, true, fin_op, stream);
  }
};

template <typename InType, typename AccType, typename OutType,
          typename OutputTile_, typename FinalLambda>
struct DistanceImpl<EucUnexpandedL1, InType, AccType, OutType, OutputTile_, FinalLambda> {
  void run(InType *x, InType *y, OutType *dist, int m, int n, int k,
                void *workspace, size_t worksize,
                FinalLambda fin_op, cudaStream_t stream) {
    l1Impl<InType, AccType, OutType, OutputTile_>(
      m, n, k, x, y, dist, fin_op, stream);
  }
};

} // anonymous namespace

/**
 * @brief Return the exact workspace size to compute the distance
 * @tparam DistanceType which distance to evaluate
 * @tparam InType input argument type
 * @tparam AccType accumulation type
 * @tparam OutType output type
 * @param x first set of points
 * @param y second set of points
 * @param m number of points in x
 * @param n number of points in y
 * @param k dimensionality
 *
 * @note If the specifed distanceType doesn't need the workspace at all, it returns 0.
 */
template <DistanceType distanceType, typename InType, typename AccType, typename OutType>
size_t getWorkspaceSize(InType* x, InType* y, int m, int n, int k) {
  size_t worksize = 0;
  constexpr bool is_allocated = distanceType <= EucExpandedCosine;
  if(is_allocated) {
    worksize += m * sizeof(AccType);
    if(x != y)
      worksize += n * sizeof(AccType);
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
 *
 * @note fin_op: This is a device lambda which is supposed to operate upon the
 * input which is AccType and returns the output in OutType. It's signature is
 * as follows:  <pre>OutType fin_op(AccType in, int g_idx);</pre>. If one needs
 * any other parameters, feel free to pass them via closure.
 */

template <DistanceType distanceType,typename InType, typename AccType, typename OutType,
          typename OutputTile_, typename FinalLambda>
void distance(InType* const x, InType* const y, OutType *dist, int m, int n, int k,
              void *workspace, size_t worksize,
              FinalLambda fin_op, cudaStream_t stream) {
  DistanceImpl<distanceType, InType, AccType, OutType, OutputTile_, FinalLambda> distImpl;
  distImpl.run(x, y, dist, m, n, k, workspace, worksize, fin_op, stream);
  CUDA_CHECK(cudaPeekAtLastError());
}

/**
 * @brief Evaluate pairwise distances for the simple use case
 * @tparam DistanceType which distance to evaluate
 * @tparam InType input argument type
 * @tparam AccType accumulation type
 * @tparam OutType output type
 * @param x first set of points
 * @param y second set of points
 * @param dist output distance matrix
 * @param m number of points in x
 * @param n number of points in y
 * @param k dimensionality
 * @param workspace temporary workspace needed for computations
 * @param worksize number of bytes of the workspace
 * @param stream cuda stream
 *
 * @note if workspace is passed as nullptr, this will return in
 *  worksize, the number of bytes of workspace required
 */
template <DistanceType distanceType, typename InType, typename AccType, typename OutType,
          typename OutputTile_>
void distance(InType* const x, InType* const y, OutType *dist, int m, int n, int k,
              void *workspace,
              size_t worksize, cudaStream_t stream) {
  auto default_fin_op =
      [] __device__(AccType d_val, int g_d_idx) { return d_val; };
  distance<distanceType, InType, AccType, OutType, OutputTile_>(
    x, y, dist, m, n, k, workspace, worksize, default_fin_op, stream);

  CUDA_CHECK(cudaPeekAtLastError());
}


/**
 * @brief Constructs an epsilon neighborhood adjacency matrix by
 * filtering the final distance by some epsilon.
 * @tparam distanceType: distance metric to compute between a and b matrices
 * @tparam T: the type of input matrices a and b
 * @tparam Lambda:
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
template<DistanceType distanceType, typename T,
                                    typename OutputTile_= OutputTile_t,
                                    typename Lambda = auto (int, bool)->void >
size_t epsilon_neighborhood(T* const a, T* const b, bool *adj, int m, int n, int k, T eps,
            void *workspace, size_t worksize, cudaStream_t stream, Lambda fused_op) {
    auto epsilon_op = [n, eps, fused_op] __device__ (T val, int global_c_idx) {
        bool acc = val <= eps;
        fused_op(global_c_idx, acc);
        return acc;
    };

    distance<distanceType, T, T, bool, OutputTile_>
            (a, b, adj, m, n, k, (void*)workspace, worksize, epsilon_op, stream);

    return worksize;
}

/**
 * @brief Constructs an epsilon neighborhood adjacency matrix by
 * filtering the final distance by some epsilon.
 * @tparam distanceType: distance metric to compute between a and b matrices
 * @tparam T: the type of input matrices a and b
 * @tparam Lambda:
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
template<DistanceType distanceType, typename T,
                                    typename OutputTile_= OutputTile_t>
size_t epsilon_neighborhood(T* const a, T* const b, bool *adj, int m, int n, int k, T eps,
            void *workspace, size_t worksize, cudaStream_t stream) {
    return epsilon_neighborhood<distanceType, T, OutputTile_>(
        a, b, adj, m, n, k, eps, workspace, worksize, stream,
        [] __device__ (int c_idx, bool acc) {}
    );
}


}; // end namespace Distance
}; // end namespace MLCommon

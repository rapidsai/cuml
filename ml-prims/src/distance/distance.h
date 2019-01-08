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

#include "cuda_utils.h"
#include "distance/cosine.h"
#include "distance/euclidean.h"
#include "distance/l1.h"
#include <cutlass/shape.h>

namespace MLCommon {
namespace Distance {

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

/**
 * @brief Evaluate pairwise distances
 * @tparam InType input argument type
 * @tparam AccType accumulation type
 * @tparam OutParams output parameter type. It could represent simple C-like struct
 * to pass extra outputs after the computation.
 * @tparam InParams input parameter type. It could represent simple C-like struct
 * to hold extra input that might be needed during the computation.
 * @param dist output parameters
 * @param x first set of points
 * @param y second set of points
 * @param m number of points in x
 * @param n number of points in y
 * @param k dimensionality
 * @param in_params extra input parameters
 * @param type which distance to evaluate
 * @param workspace temporary workspace needed for computations
 * @param worksize number of bytes of the workspace
 * @param stream cuda stream
 *
 * @note if workspace is passed as nullptr, this will return in
 *  worksize, the number of bytes of workspace required
 */
template <typename InType, typename AccType, typename InParams,
          typename OutParams, typename OutputTile_, typename FinalLambda>
void distance(InType* x, InType* y, int m, int n, int k,
              InParams const& in_params, OutParams& out_params, DistanceType type,
              void* workspace, size_t& worksize,
              FinalLambda fin_op,
              cudaStream_t stream=0) {

    if(workspace == nullptr && type <= EucExpandedCosine) {
        worksize = m * sizeof(AccType);
        if(x != y)
            worksize += n * sizeof(AccType);
        return;
    }

    ///@todo: implement the distance matrix computation here
    switch(type) {
    case EucExpandedL2:
        euclideanAlgo1<InType, AccType, AccType, OutputTile_>
          (m, n, k, x, y, out_params.dist, out_params.dist, (AccType)1, (AccType)0,
           false, in_params, out_params,
           (AccType*)workspace, worksize,
           fin_op);
        break;
    case EucExpandedL2Sqrt:
        euclideanAlgo1<InType, AccType, AccType, OutputTile_>
          (m, n, k, x, y, out_params.dist, out_params.dist, (AccType)1, (AccType)0,
           true, in_params, out_params,
           (AccType*)workspace, worksize,
           fin_op);
        break;
    case EucUnexpandedL2:
        euclideanAlgo2<InType, AccType, AccType, OutputTile_>
        (m, n, k, x, y, out_params.dist, out_params.dist, (AccType)1, (AccType)0);
        break;
    case EucUnexpandedL2Sqrt:
        euclideanAlgo2<InType, AccType, AccType, OutputTile_>
        (m, n, k, x, y, out_params.dist, out_params.dist, (AccType)1, (AccType)0, true);
        break;
    case EucUnexpandedL1:
        l1Impl<InType, AccType, AccType, OutputTile_>
          (m, n, k, x, y, out_params.dist, out_params.dist, (AccType)1, (AccType)0, stream);
        break;
    // case EucExpandedCosine:
    //     cosineAlgo1<InType, AccType, AccType, OutputTile_>
    //       (m, n, k, x, y, out_params.dist, out_params.dist, (AccType)1, (AccType)0,
    //       (AccType*)workspace, worksize, stream);
    //     break;
    default:
        ASSERT(false, "Invalid DistanceType '%d'!", type);
    };
}

template <typename InType, typename AccType, typename InParams,
          typename OutParams, typename OutputTile_>
void distance(InType* x, InType* y, int m, int n, int k,
              InParams const& in_params, OutParams& out_params, DistanceType type,
              void* workspace, size_t& worksize,
              cudaStream_t stream=0) {
  auto default_fin_op = [] __device__
                        (AccType d_val, int g_d_idx,
                            const InParams& in_params, OutParams& out_params) {
                          return (InType)d_val;
                        };
  distance<InType, AccType, InParams, OutParams,
           OutputTile_>
      (x, y, m, n, k,
       in_params, out_params, type,
       workspace, worksize,
       default_fin_op,
       stream);
}

}; // end namespace Distance
}; // end namespace MLCommon


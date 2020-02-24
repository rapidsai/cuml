/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include "distance.h"
#include <linalg/contractions.cuh>
#include <common/device_utils.cuh>

namespace MLCommon {
namespace Distance {

/**
 * @defgroup EpsNeigh Epsilon Neighborhood computation
 * @{
 * @brief Constructs an epsilon neighborhood adjacency matrix by filtering the
 *        final distance by some epsilon.
 *
 * @tparam distanceType distance metric to compute between a and b matrices
 * @tparam T            the type of input matrices a and b
 * @tparam Lambda       Lambda function
 * @tparam Index_       Index type
 * @tparam OutputTile_  output tile size per thread
 *
 * @param a         first matrix [row-major] [on device] [dim = m x k]
 * @param b         second matrix [row-major] [on device] [dim = n x k]
 * @param adj       a boolean output adjacency matrix [row-major] [on device]
 *                  [dim = m x n]
 * @param m         number of points in a
 * @param n         number of points in b
 * @param k         dimensionality
 * @param eps       epsilon value to use as a filter for neighborhood
 *                  construction. It is important to note that if the distance
 *                  type returns a squared variant for efficiency, epsilon will
 *                  need to be squared as well.
 * @param workspace temporary workspace needed for computations
 * @param worksize  number of bytes of the workspace
 * @param stream    cuda stream
 * @param fused_op  optional functor taking the output index into c
 *                  and a boolean denoting whether or not the inputs are part of
 *                  the epsilon neighborhood.
 * @return          the workspace size in bytes
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
/** @} */

template <typename DataT, typename IdxT, typename Policy,
          typename BaseClass = LinAlg::Contractions_NT<DataT, IdxT, Policy>>
struct EpsUnexpL2SqNeighborhood : public BaseClass {
 private:
  typedef Policy P;

  bool* adj;
  DataT eps;
  IdxT* vd;

  char* smem;  // for final reductions

  DataT acc[P::AccRowsPerTh][P::AccColsPerTh];

 public:
  DI EpsUnexpL2SqNeighborhood(bool* _adj, IdxT* _vd, const DataT* _x,
                              const DataT* _y, IdxT _m, IdxT _n, IdxT _k,
                              DataT _eps, char* _smem)
    : BaseClass(_x, _y, _m, _n, _k, _smem), adj(_adj), eps(_eps), vd(_vd),
      smem(_smem) {
  }

  DI void run() {
    prolog();
    loop();
    __syncthreads();  // so that we can safely reuse smem
    epilog();
  }

 private:
  DI void prolog() {
    this->ldgsts(0);
    this->pageWr ^= 1;
#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < P::AccColsPerTh; ++j) {
        acc[i][j] = BaseClass::Zero;
      }
    }
    __syncthreads();
  }

  DI void loop() {
    for (int kidx = P::Kblk; kidx < this->k; kidx += P::Kblk) {
      this->ldgsts(kidx);
      accumulate();  // on the previous k-block
      __syncthreads();
      this->pageWr ^= 1;
      this->pageRd ^= 1;
    }
    accumulate();  // last iteration
  }

  DI void epilog() {
    IdxT startx = blockIdx.x * P::Mblk + this->accrowid;
    IdxT starty = blockIdx.y * P::Nblk + this->acccolid;
    auto lid = laneId();
    IdxT sums[P::AccColsPerTh];
#pragma unroll
    for (int j = 0; j < P::AccColsPerTh; ++j) {
      sums[j] = 0;
    }
#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
      auto xid = startx + i * P::AccThRows;
#pragma unroll
      for (int j = 0; j < P::AccColsPerTh; ++j) {
        auto yid = starty + j * P::AccThCols;
        auto is_neigh = acc[i][j] <= eps;
        ///@todo: fix uncoalesced writes using shared mem
        if (xid < this->m && yid < this->n) {
          adj[xid * this->n + yid] = is_neigh;
          sums[j] += is_neigh;
        }
      }
    }
    // perform reduction of adjacency values to compute vertex degrees
    if (vd != nullptr) {
      updateVertexDegree(sums);
    }
  }

  DI void accumulate() {
#pragma unroll
    for (int ki = 0; ki < P::Kblk; ki += P::Veclen) {
      this->ldsXY(ki);
#pragma unroll
      for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
        for (int j = 0; j < P::AccColsPerTh; ++j) {
#pragma unroll
          for (int v = 0; v < P::Veclen; ++v) {
            auto diff = this->regx[i][v] - this->regy[j][v];
            acc[i][j] += diff * diff;
          }
        }
      }
    }
  }

  DI void updateVertexDegree(IdxT (&sums)[P::AccColsPerTh]) {
    __syncthreads();
    int gid = threadIdx.x / P::AccThCols;
    int lid = threadIdx.x % P::AccThCols;
    auto cidx = IdxT(blockIdx.y) * P::Nblk + lid;
    IdxT totalSum = 0;
    // update the individual vertex degrees
#pragma unroll
    for (int i = 0; i < P::AccColsPerTh; ++i) {
      sums[i] = batchedBlockReduce<IdxT, P::AccThCols>(sums[i], smem);
      auto cid = cidx + i * P::AccThCols;
      if (gid == 0 && cid < this->n) {
        if (sizeof(IdxT) == 4) {
          myAtomicAdd((unsigned*)(vd + cid), sums[i]);
        } else if (sizeof(IdxT) == 8) {
          myAtomicAdd((unsigned long long*)(vd + cid), sums[i]);
        }
        totalSum += sums[i];
      }
      __syncthreads();  // for safe smem reuse
    }
    // update the total edge count
    totalSum = blockReduce<IdxT>(totalSum, smem);
    if (threadIdx.x == 0) {
      if (sizeof(IdxT) == 4) {
        myAtomicAdd((unsigned*)(vd + this->n), totalSum);
      } else if (sizeof(IdxT) == 8) {
        myAtomicAdd((unsigned long long*)(vd + this->n), totalSum);
      }
    }
  }
};  // struct EpsUnexpL2SqNeighborhood

template <typename DataT, typename IdxT, typename Policy>
__global__ __launch_bounds__(Policy::Nthreads, 2) void epsUnexpL2SqNeighKernel(
  bool* adj, IdxT* vd, const DataT* x, const DataT* y, IdxT m, IdxT n, IdxT k,
  DataT eps) {
  extern __shared__ char smem[];
  EpsUnexpL2SqNeighborhood<DataT, IdxT, Policy> obj(
    adj, vd, x, y, m, n, k, eps, smem);
  obj.run();
}

template <typename DataT, typename IdxT, int VecLen>
void epsUnexpL2SqNeighImpl(bool* adj, IdxT* vd, const DataT* x, const DataT* y,
                           IdxT m, IdxT n, IdxT k, DataT eps,
                           cudaStream_t stream) {
  typedef typename LinAlg::Policy4x4<DataT, VecLen>::Policy Policy;
  dim3 grid(ceildiv<int>(m, Policy::Mblk), ceildiv<int>(n, Policy::Nblk));
  dim3 blk(Policy::Nthreads);
  epsUnexpL2SqNeighKernel<DataT, IdxT, Policy>
    <<<grid, blk, Policy::SmemSize, stream>>>(adj, vd, x, y, m, n, k, eps);
  CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Computes epsilon neighborhood for the L2-Squared distance metric
 *
 * @tparam DataT   IO and math type
 * @tparam IdxT    Index type
 *
 * @param[out] adj    adjacency matrix [row-major] [on device] [dim = m x n]
 * @param[out] vd     vertex degree array [on device] [len = m + 1]
 *                    `vd + m` stores the total number of edges in the adjacency
 *                    matrix. Pass a nullptr if you don't need this info.
 * @param[in]  x      first matrix [row-major] [on device] [dim = m x k]
 * @param[in]  y      second matrix [row-major] [on device] [dim = n x k]
 * @param[in]  eps    defines epsilon neighborhood radius (should be passed as
 *                    squared as we compute L2-squared distance in this method)
 * @param[in]  fop    device lambda to do any other custom functions
 * @param[in]  stream cuda stream
 */
template <typename DataT, typename IdxT>
void epsUnexpL2SqNeighborhood(bool* adj, IdxT* vd, const DataT* x,
                              const DataT* y, IdxT m, IdxT n, IdxT k, DataT eps,
                              cudaStream_t stream) {
  size_t bytes = sizeof(DataT) * k;
  if (16 % sizeof(DataT) == 0 && bytes % 16 == 0) {
    epsUnexpL2SqNeighImpl<DataT, IdxT, 16 / sizeof(DataT)>(
      adj, vd, x, y, m, n, k, eps, stream);
  } else if (8 % sizeof(DataT) == 0 && bytes % 8 == 0) {
    epsUnexpL2SqNeighImpl<DataT, IdxT, 8 / sizeof(DataT)>(
      adj, vd, x, y, m, n, k, eps, stream);
  } else {
    epsUnexpL2SqNeighImpl<DataT, IdxT, 1>(
      adj, vd, x, y, m, n, k, eps, stream);
  }
}

}  // namespace Distance
}  // namespace MLCommon

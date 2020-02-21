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

template <typename DataT, typename IdxT, typename Policy, typename FusedOp,
          typename BaseClass = LinAlg::Contractions_NT<DataT, IdxT, Policy>>
struct EpsUnexpL2SqNeighborhood : public BaseClass {
 private:
  typedef Policy P;

  bool* adj;
  DataT eps;

  FusedOp fusedOp;

  DataT acc[P::AccRowsPerTh][P::AccColsPerTh];

 public:
  DI EpsUnexpL2SqNeighborhood(bool* _adj, const DataT* _x, const DataT* _y,
                              IdxT _m, IdxT _n, IdxT _k, DataT _eps,
                              FusedOp fop, char* _smem)
    : BaseClass(_x, _y, _m, _n, _k, _smem), adj(_adj), eps(_eps), fusedOp(fop) {
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
    IdxT startx = blockIdx.x * P::Mblk;
    IdxT starty = blockIdx.y * P::Nblk;
#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
      auto xid = startx + i * P::AccThRows;
#pragma unroll
      for (int j = 0; j < P::AccColsPerTh; ++j) {
        auto is_neigh = acc[i][j] <= eps;
        auto yid = starty + j * P::AccThCols;
        ///@todo: fix uncoalesced writes using shared mem
        if (xid < this->m && yid < this->n) {
          adj[xid * this->n + yid] = is_neigh;
          fusedOp(is_neigh, xid, yid);
        }
      }
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
};  // struct EpsUnexpL2SqNeighborhood

template <typename DataT, typename IdxT, typename Policy, typename FusedOp>
__global__ __launch_bounds__(Policy::Nthreads, 2) void epsUnexpL2SqNeighKernel(
  bool* adj, const DataT* x, const DataT* y, IdxT m, IdxT n, IdxT k, DataT eps,
  FusedOp fop) {
  extern __shared__ char smem[];
  EpsUnexpL2SqNeighborhood<DataT, IdxT, Policy, FusedOp> obj(
    adj, x, y, m, n, k, eps, fop, smem);
  obj.run();
}

template <typename DataT, typename IdxT, int VecLen, typename FusedOp>
void epsUnexpL2SqNeighImpl(bool* adj, const DataT* x, const DataT* y, IdxT m,
                           IdxT n, IdxT k, DataT eps, FusedOp fop,
                           cudaStream_t stream) {
  typedef typename LinAlg::Policy4x4<DataT, VecLen>::Policy Policy;
  dim3 grid(ceildiv<int>(m, Policy::Mblk), ceildiv<int>(n, Policy::Nblk));
  dim3 blk(Policy::Nthreads);
  epsUnexpL2SqNeighKernel<DataT, IdxT, Policy, FusedOp>
    <<<grid, blk, Policy::SmemSize, stream>>>(adj, x, y, m, n, k, eps, fop);
  CUDA_CHECK(cudaGetLastError());
}

template <typename DataT, typename IdxT, typename FusedOp>
void epsUnexpL2SqNeighborhood(bool* adj, const DataT* x, const DataT* y, IdxT m,
                              IdxT n, IdxT k, DataT eps, FusedOp fop,
                              cudaStream_t stream) {
  size_t bytes = sizeof(DataT) * k;
  if (16 % sizeof(DataT) == 0 && bytes % 16 == 0) {
    epsUnexpL2SqNeighImpl<DataT, IdxT, 16 / sizeof(DataT), FusedOp>(
      adj, x, y, m, n, k, eps, fop, stream);
  } else if (8 % sizeof(DataT) == 0 && bytes % 8 == 0) {
    epsUnexpL2SqNeighImpl<DataT, IdxT, 8 / sizeof(DataT), FusedOp>(
      adj, x, y, m, n, k, eps, fop, stream);
  } else {
    epsUnexpL2SqNeighImpl<DataT, IdxT, 1, FusedOp>(
      adj, x, y, m, n, k, eps, fop, stream);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace Distance
}  // namespace MLCommon

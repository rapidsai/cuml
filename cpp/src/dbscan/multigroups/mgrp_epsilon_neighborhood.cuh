/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

#include <cuda_runtime.h>
#include <math.h>
#include <raft/spatial/knn/epsilon_neighborhood.cuh>

#include "mgrp_accessor.cuh"

namespace ML {
namespace Dbscan {
namespace Multigroups {
namespace VertexDeg {
namespace EpsNeighborhood {

/**
 * The implementation is based on
 * https://github.com/rapidsai/raft/blob/branch-23.04/cpp/include/raft/spatial/knn/detail/epsilon_neighborhood.cuh
 */
template <typename DataT,
          typename IdxT,
          typename Policy,
          typename BaseClass = raft::linalg::Contractions_NT<DataT, IdxT, Policy>>
struct MgrpEpsUnexpL2SqNeighborhood : public BaseClass {
 private:
  typedef Policy P;

  IdxT data_start_id;
  IdxT adj_stride;

  bool* adj;
  DataT eps;

  IdxT* vd;
  IdxT* vd_group;
  IdxT* vd_all;

  char* smem;  // for final reductions

  DataT acc[P::AccRowsPerTh][P::AccColsPerTh];

 public:
  DI MgrpEpsUnexpL2SqNeighborhood(bool* _adj,
                                  IdxT* _vd,
                                  IdxT* _vd_group,
                                  IdxT* _vd_all,
                                  const DataT* _x,
                                  const DataT* _y,
                                  IdxT _m,
                                  IdxT _n,
                                  IdxT _k,
                                  IdxT _data_start_id,
                                  IdxT _adj_stride,
                                  DataT _eps,
                                  char* _smem)
    : BaseClass(_x, _y, _m, _n, _k, _smem),
      adj(_adj),
      eps(_eps),
      data_start_id(_data_start_id),
      adj_stride(_adj_stride),
      vd(_vd),
      vd_group(_vd_group),
      vd_all(_vd_all),
      smem(_smem)
  {
  }

  DI void run()
  {
    prolog();
    loop();
    epilog();
  }

 private:
  DI void prolog()
  {
    this->ldgXY(IdxT(blockIdx.x) * P::Mblk, IdxT(blockIdx.y) * P::Nblk, 0);
#pragma unroll
    for (int i = 0; i < P::AccRowsPerTh; ++i) {
#pragma unroll
      for (int j = 0; j < P::AccColsPerTh; ++j) {
        acc[i][j] = BaseClass::Zero;
      }
    }
    this->stsXY();
    __syncthreads();
    this->switch_write_buffer();
  }

  DI void loop()
  {
    for (int kidx = P::Kblk; kidx < this->k; kidx += P::Kblk) {
      this->ldgXY(IdxT(blockIdx.x) * P::Mblk, IdxT(blockIdx.y) * P::Nblk, kidx);
      accumulate();  // on the previous k-block
      this->stsXY();
      __syncthreads();
      this->switch_write_buffer();
      this->switch_read_buffer();
    }
    accumulate();  // last iteration
  }

  DI void epilog()
  {
    IdxT startx = blockIdx.x * P::Mblk + this->accrowid;
    IdxT starty = blockIdx.y * P::Nblk + this->acccolid;
    auto lid    = raft::laneId();
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
        auto yid      = starty + j * P::AccThCols;
        auto adj_xid  = xid;
        auto is_neigh = acc[i][j] <= eps;
        ///@todo: fix uncoalesced writes using shared mem
        if (xid < this->m && yid < this->n) {
          // adj[adj_xid * this->n + yid] = is_neigh;
          adj[yid * this->adj_stride + adj_xid] = is_neigh;
          sums[j] += is_neigh;
        }
      }
    }
    // perform reduction of adjacency values to compute vertex degrees
    if (vd == nullptr || vd_group == nullptr || vd_all == nullptr) return;
    updateVertexDegree(sums);
  }

  DI void accumulate()
  {
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

  DI void updateVertexDegree(IdxT (&sums)[P::AccColsPerTh])
  {
    __syncthreads();  // so that we can safely reuse smem
    int gid       = threadIdx.x / P::AccThCols;
    int lid       = threadIdx.x % P::AccThCols;
    auto cidx     = IdxT(blockIdx.y) * P::Nblk + lid;
    IdxT totalSum = 0;
    // update the individual vertex degrees
#pragma unroll
    for (int i = 0; i < P::AccColsPerTh; ++i) {
      sums[i]  = raft::batchedBlockReduce<IdxT, P::AccThCols>(sums[i], smem);
      auto cid = cidx + i * P::AccThCols;
      if (gid == 0 && cid < this->n) {
        atomicUpdate(cid, sums[i]);
        totalSum += sums[i];
      }
      __syncthreads();  // for safe smem reuse
    }
    // update the total edge count
    totalSum = raft::blockReduce<IdxT>(totalSum, smem);
    if (threadIdx.x == 0) {
      atomicUpdate(vd_group, totalSum);
      atomicUpdate(vd_all, totalSum);
    }
  }

  DI void atomicUpdate(IdxT addrId, IdxT val)
  {
    if (sizeof(IdxT) == 4) {
      raft::myAtomicAdd<unsigned>((unsigned*)(vd + addrId), val);
    } else if (sizeof(IdxT) == 8) {
      raft::myAtomicAdd<unsigned long long>((unsigned long long*)(vd + addrId), val);
    }
  }

  DI void atomicUpdate(IdxT* addr, IdxT val)
  {
    if (sizeof(IdxT) == 4) {
      raft::myAtomicAdd<unsigned>((unsigned*)(addr), val);
    } else if (sizeof(IdxT) == 8) {
      raft::myAtomicAdd<unsigned long long>((unsigned long long*)(addr), val);
    }
  }
};

template <typename DataT, typename IdxT, typename Policy>
__global__ __launch_bounds__(Policy::Nthreads, 2) void MultiGroupEpsUnexpL2SqNeighKernel(
  Metadata::AdjGraphAccessor<bool, IdxT> adj_ac,
  Metadata::VertexDegAccessor<IdxT, IdxT> vd_ac,
  const Metadata::PointAccessor<DataT, IdxT> x_ac,
  const Metadata::PointAccessor<DataT, IdxT> y_ac,
  bool calc_vd,
  DataT* eps)
{
  extern __shared__ char smem[];
  IdxT blk_x    = blockIdx.x;
  IdxT blk_y    = blockIdx.y;
  IdxT group_id = blockIdx.z;

  IdxT group_start_row  = x_ac.row_start_ids[group_id];
  IdxT group_valid_rows = x_ac.n_rows_ptr[group_id];

  IdxT n_groups = x_ac.n_groups;
  IdxT m        = group_valid_rows;
  IdxT n        = group_valid_rows;
  IdxT k        = x_ac.feat_size;

  if (group_id >= n_groups || blk_x >= raft::ceildiv<int>(m, Policy::Mblk) ||
      blk_y >= raft::ceildiv<int>(n, Policy::Nblk))
    return;

  bool* adj      = adj_ac.adj + adj_ac.adj_group_offset[group_id];
  IdxT* vd       = (calc_vd) ? vd_ac.vd + group_start_row : nullptr;
  IdxT* vd_group = (calc_vd) ? vd_ac.vd_group + group_id : nullptr;
  IdxT* vd_all   = (calc_vd) ? vd_ac.vd_all : nullptr;
  const DataT* x = x_ac.pts + group_start_row * k;
  const DataT* y = y_ac.pts + group_start_row * k;
  DataT eps_val  = eps[group_id];

  MgrpEpsUnexpL2SqNeighborhood<DataT, IdxT, Policy> obj(adj,
                                                        vd,
                                                        vd_group,
                                                        vd_all,
                                                        x,
                                                        y,
                                                        m,
                                                        n,
                                                        k,
                                                        group_start_row,
                                                        adj_ac.adj_col_stride[group_id],
                                                        eps_val,
                                                        smem);
  obj.run();
}

template <typename DataT, typename IdxT, int VecLen>
static void MultiGroupEpsUnexpL2SqNeighImpl(Metadata::AdjGraphAccessor<bool, IdxT>& adj_ac,
                                            Metadata::VertexDegAccessor<IdxT, IdxT>& vd_ac,
                                            const Metadata::PointAccessor<DataT, IdxT>& x_ac,
                                            const Metadata::PointAccessor<DataT, IdxT>& y_ac,
                                            DataT* eps,
                                            bool calc_vd,
                                            cudaStream_t stream)
{
  IdxT m = x_ac.max_rows;
  IdxT n = x_ac.max_rows;
  typedef typename raft::linalg::Policy4x4<DataT, VecLen>::Policy Policy;
  dim3 grid(
    raft::ceildiv<int>(m, Policy::Mblk), raft::ceildiv<int>(n, Policy::Nblk), x_ac.n_groups);
  dim3 blk(Policy::Nthreads);
  MultiGroupEpsUnexpL2SqNeighKernel<DataT, IdxT, Policy>
    <<<grid, blk, Policy::SmemSize, stream>>>(adj_ac, vd_ac, x_ac, y_ac, calc_vd, eps);
  RAFT_CUDA_TRY(cudaGetLastError());
}

template <typename DataT, typename IdxT = int>
void MultiGroupEpsUnexpL2SqNeighborhood(Metadata::AdjGraphAccessor<bool, IdxT>& adj_ac,
                                        Metadata::VertexDegAccessor<IdxT, IdxT>& vd_ac,
                                        const Metadata::PointAccessor<DataT, IdxT>& x_ac,
                                        const Metadata::PointAccessor<DataT, IdxT>& y_ac,
                                        DataT* eps,
                                        bool calc_vd,
                                        cudaStream_t stream)
{
  ASSERT(sizeof(IdxT) == 4 || sizeof(IdxT) == 8, "IdxT should be 4 or 8 bytes");
  IdxT k       = x_ac.feat_size;
  size_t bytes = sizeof(DataT) * k;
  if (16 % sizeof(DataT) == 0 && bytes % 16 == 0) {
    MultiGroupEpsUnexpL2SqNeighImpl<DataT, IdxT, 16 / sizeof(DataT)>(
      adj_ac, vd_ac, x_ac, y_ac, eps, calc_vd, stream);
  } else if (8 % sizeof(DataT) == 0 && bytes % 8 == 0) {
    MultiGroupEpsUnexpL2SqNeighImpl<DataT, IdxT, 8 / sizeof(DataT)>(
      adj_ac, vd_ac, x_ac, y_ac, eps, calc_vd, stream);
  } else {
    MultiGroupEpsUnexpL2SqNeighImpl<DataT, IdxT, 1>(
      adj_ac, vd_ac, x_ac, y_ac, eps, calc_vd, stream);
  }
}

}  // namespace EpsNeighborhood
}  // end namespace VertexDeg
}  // namespace Multigroups
}  // end namespace Dbscan
}  // namespace ML
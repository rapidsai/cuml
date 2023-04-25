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

#include "mgrp_accessor.cuh"

#include <cub/cub.cuh>
#include <raft/common/nvtx.hpp>
#include <raft/core/operators.hpp>
#include <raft/util/cuda_utils.cuh>

namespace ML {
namespace Dbscan {
namespace Multigroups {
namespace VertexDeg {
namespace Reduce {

/**
 * The implementation is based on
 * https://github.com/rapidsai/raft/blob/branch-23.06/cpp/include/raft/linalg/detail/coalesced_reduction.cuh
 */
template <int warpSize, int rpb>
struct ReductionThinPolicy {
  static constexpr int LogicalWarpSize = warpSize;
  static constexpr int RowsPerBlock    = rpb;
  static constexpr int ThreadsPerBlock = LogicalWarpSize * RowsPerBlock;
};

template <typename Policy,
          typename InType,
          typename OutType,
          typename IdxType,
          typename MainLambda,
          typename ReduceLambda,
          typename FinalLambda>
__global__ void __launch_bounds__(Policy::ThreadsPerBlock)
  coalescedReductionThinKernel(OutType* dots,
                               const InType* data,
                               IdxType n_groups,
                               const IdxType* dev_d_ptr,
                               const IdxType* dev_n_ptr,
                               const IdxType* dots_offsets,
                               const std::size_t* data_offsets,
                               OutType init,
                               MainLambda main_op,
                               ReduceLambda reduce_op,
                               FinalLambda final_op,
                               bool inplace = false)
{
  IdxType group_id = blockIdx.z * blockDim.z + threadIdx.z;
  if (group_id >= n_groups) return;

  OutType* dots_base      = dots + dots_offsets[group_id];
  const InType* data_base = data + data_offsets[group_id];
  IdxType D               = dev_d_ptr[group_id];
  IdxType N               = dev_n_ptr[group_id];

  IdxType i = threadIdx.y + (Policy::RowsPerBlock * static_cast<IdxType>(blockIdx.x));
  if (i >= N) return;

  OutType acc = init;
  for (IdxType j = threadIdx.x; j < D; j += Policy::LogicalWarpSize) {
    acc = reduce_op(acc, main_op(data_base[j + (D * i)], j));
  }
  acc = raft::logicalWarpReduce<Policy::LogicalWarpSize>(acc, reduce_op);
  if (threadIdx.x == 0) {
    if (inplace) {
      dots_base[i] = final_op(reduce_op(dots_base[i], acc));
    } else {
      dots_base[i] = final_op(acc);
    }
  }
}

template <typename Policy,
          typename InType,
          typename OutType      = InType,
          typename IdxType      = int,
          typename MainLambda   = raft::identity_op,
          typename ReduceLambda = raft::add_op,
          typename FinalLambda  = raft::identity_op>
void coalescedReductionThin(OutType* dots,
                            const InType* data,
                            IdxType n_groups,
                            IdxType N,
                            const IdxType* dev_d_ptr,
                            const IdxType* dev_n_ptr,
                            const IdxType* dots_offsets,
                            const std::size_t* data_offsets,
                            OutType init,
                            cudaStream_t stream,
                            bool inplace           = false,
                            MainLambda main_op     = raft::identity_op(),
                            ReduceLambda reduce_op = raft::add_op(),
                            FinalLambda final_op   = raft::identity_op())
{
  raft::common::nvtx::range<raft::common::nvtx::domain::raft> fun_scope(
    "coalescedReductionThin<%d,%d>", Policy::LogicalWarpSize, Policy::RowsPerBlock);
  dim3 threads(Policy::LogicalWarpSize, Policy::RowsPerBlock, 1);
  dim3 blocks(raft::ceildiv<IdxType>(N, Policy::RowsPerBlock), 1, n_groups);
  coalescedReductionThinKernel<Policy><<<blocks, threads, 0, stream>>>(dots,
                                                                       data,
                                                                       n_groups,
                                                                       dev_d_ptr,
                                                                       dev_n_ptr,
                                                                       dots_offsets,
                                                                       data_offsets,
                                                                       init,
                                                                       main_op,
                                                                       reduce_op,
                                                                       final_op,
                                                                       inplace);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename InType,
          typename OutType      = InType,
          typename IdxType      = int,
          typename MainLambda   = raft::identity_op,
          typename ReduceLambda = raft::add_op,
          typename FinalLambda  = raft::identity_op>
void coalescedReductionThinDispatcher(OutType* dots,
                                      const InType* data,
                                      IdxType n_groups,
                                      IdxType D,
                                      IdxType N,
                                      const IdxType* dev_d_ptr,
                                      const IdxType* dev_n_ptr,
                                      const IdxType* dots_offsets,
                                      const std::size_t* data_offsets,
                                      OutType init,
                                      cudaStream_t stream,
                                      bool inplace           = false,
                                      MainLambda main_op     = raft::identity_op(),
                                      ReduceLambda reduce_op = raft::add_op(),
                                      FinalLambda final_op   = raft::identity_op())
{
  if (D <= IdxType(2)) {
    coalescedReductionThin<ReductionThinPolicy<2, 64>>(dots,
                                                       data,
                                                       n_groups,
                                                       N,
                                                       dev_d_ptr,
                                                       dev_n_ptr,
                                                       dots_offsets,
                                                       data_offsets,
                                                       init,
                                                       stream,
                                                       inplace,
                                                       main_op,
                                                       reduce_op,
                                                       final_op);
  } else if (D <= IdxType(4)) {
    coalescedReductionThin<ReductionThinPolicy<4, 32>>(dots,
                                                       data,
                                                       n_groups,
                                                       N,
                                                       dev_d_ptr,
                                                       dev_n_ptr,
                                                       dots_offsets,
                                                       data_offsets,
                                                       init,
                                                       stream,
                                                       inplace,
                                                       main_op,
                                                       reduce_op,
                                                       final_op);
  } else if (D <= IdxType(8)) {
    coalescedReductionThin<ReductionThinPolicy<8, 16>>(dots,
                                                       data,
                                                       n_groups,
                                                       N,
                                                       dev_d_ptr,
                                                       dev_n_ptr,
                                                       dots_offsets,
                                                       data_offsets,
                                                       init,
                                                       stream,
                                                       inplace,
                                                       main_op,
                                                       reduce_op,
                                                       final_op);
  } else if (D <= IdxType(16)) {
    coalescedReductionThin<ReductionThinPolicy<16, 8>>(dots,
                                                       data,
                                                       n_groups,
                                                       N,
                                                       dev_d_ptr,
                                                       dev_n_ptr,
                                                       dots_offsets,
                                                       data_offsets,
                                                       init,
                                                       stream,
                                                       inplace,
                                                       main_op,
                                                       reduce_op,
                                                       final_op);
  } else {
    coalescedReductionThin<ReductionThinPolicy<32, 4>>(dots,
                                                       data,
                                                       n_groups,
                                                       N,
                                                       dev_d_ptr,
                                                       dev_n_ptr,
                                                       dots_offsets,
                                                       data_offsets,
                                                       init,
                                                       stream,
                                                       inplace,
                                                       main_op,
                                                       reduce_op,
                                                       final_op);
  }
}

template <int TPB,
          typename InType,
          typename OutType,
          typename IdxType,
          typename MainLambda,
          typename ReduceLambda,
          typename FinalLambda>
__global__ void __launch_bounds__(TPB)
  coalescedReductionMediumKernel(OutType* dots,
                                 const InType* data,
                                 IdxType n_groups,
                                 const IdxType* dev_d_ptr,
                                 const IdxType* dev_n_ptr,
                                 const IdxType* dots_offsets,
                                 const std::size_t* data_offsets,
                                 OutType init,
                                 MainLambda main_op,
                                 ReduceLambda reduce_op,
                                 FinalLambda final_op,
                                 bool inplace = false)
{
  typedef cub::BlockReduce<OutType, TPB, cub::BLOCK_REDUCE_RAKING> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  IdxType group_id = blockIdx.z * blockDim.z + threadIdx.z;
  if (group_id >= n_groups) return;

  OutType* dots_base      = dots + dots_offsets[group_id];
  const InType* data_base = data + data_offsets[group_id];
  IdxType D               = dev_d_ptr[group_id];
  IdxType N               = dev_n_ptr[group_id];

  OutType thread_data = init;
  IdxType rowStart    = blockIdx.x * D;
  for (IdxType i = threadIdx.x; i < D; i += TPB) {
    IdxType idx = rowStart + i;
    thread_data = reduce_op(thread_data, main_op(data_base[idx], i));
  }
  OutType acc = BlockReduce(temp_storage).Reduce(thread_data, reduce_op);
  if (threadIdx.x == 0) {
    if (inplace) {
      dots_base[blockIdx.x] = final_op(reduce_op(dots_base[blockIdx.x], acc));
    } else {
      dots_base[blockIdx.x] = final_op(acc);
    }
  }
}

template <int TPB,
          typename InType,
          typename OutType      = InType,
          typename IdxType      = int,
          typename MainLambda   = raft::identity_op,
          typename ReduceLambda = raft::add_op,
          typename FinalLambda  = raft::identity_op>
void coalescedReductionMedium(OutType* dots,
                              const InType* data,
                              IdxType n_groups,
                              IdxType N,
                              const IdxType* dev_d_ptr,
                              const IdxType* dev_n_ptr,
                              const IdxType* dots_offsets,
                              const std::size_t* data_offsets,
                              OutType init,
                              cudaStream_t stream,
                              bool inplace           = false,
                              MainLambda main_op     = raft::identity_op(),
                              ReduceLambda reduce_op = raft::add_op(),
                              FinalLambda final_op   = raft::identity_op())
{
  raft::common::nvtx::range<raft::common::nvtx::domain::raft> fun_scope(
    "coalescedReductionMedium<%d>", TPB);
  dim3 threads(TPB, 1, 1);
  dim3 blocks(N, 1, n_groups);
  coalescedReductionMediumKernel<TPB><<<blocks, threads, 0, stream>>>(dots,
                                                                      data,
                                                                      n_groups,
                                                                      dev_d_ptr,
                                                                      dev_n_ptr,
                                                                      dots_offsets,
                                                                      data_offsets,
                                                                      init,
                                                                      main_op,
                                                                      reduce_op,
                                                                      final_op,
                                                                      inplace);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename InType,
          typename OutType      = InType,
          typename IdxType      = int,
          typename MainLambda   = raft::identity_op,
          typename ReduceLambda = raft::add_op,
          typename FinalLambda  = raft::identity_op>
void coalescedReductionMediumDispatcher(OutType* dots,
                                        const InType* data,
                                        IdxType n_groups,
                                        IdxType D,
                                        IdxType N,
                                        const IdxType* dev_d_ptr,
                                        const IdxType* dev_n_ptr,
                                        const IdxType* dots_offsets,
                                        const std::size_t* data_offsets,
                                        OutType init,
                                        cudaStream_t stream,
                                        bool inplace           = false,
                                        MainLambda main_op     = raft::identity_op(),
                                        ReduceLambda reduce_op = raft::add_op(),
                                        FinalLambda final_op   = raft::identity_op())
{
  // Note: for now, this kernel is only used when D > 256. If this changes in the future, use
  // smaller block sizes when relevant.
  coalescedReductionMedium<256>(dots,
                                data,
                                n_groups,
                                N,
                                dev_d_ptr,
                                dev_n_ptr,
                                dots_offsets,
                                data_offsets,
                                init,
                                stream,
                                inplace,
                                main_op,
                                reduce_op,
                                final_op);
}

// Primitive to perform reductions along the coalesced dimension of the matrix, i.e. reduce along
// rows for row major or reduce along columns for column major layout. Can do an inplace reduction
// adding to original values of dots if requested.
template <typename InType,
          typename OutType      = InType,
          typename IdxType      = int,
          typename MainLambda   = raft::identity_op,
          typename ReduceLambda = raft::add_op,
          typename FinalLambda  = raft::identity_op>
void MultiGroupCoalescedReduction(OutType* mgrp_dots,
                                  const InType* mgrp_data,
                                  IdxType n_groups,
                                  IdxType D,
                                  IdxType N,
                                  const IdxType* dev_d_ptr,
                                  const IdxType* dev_n_ptr,
                                  const IdxType* dots_offsets,
                                  const std::size_t* data_offsets,
                                  OutType init,
                                  cudaStream_t stream,
                                  bool inplace           = false,
                                  MainLambda main_op     = raft::identity_op(),
                                  ReduceLambda reduce_op = raft::add_op(),
                                  FinalLambda final_op   = raft::identity_op())
{
  /* The primitive selects one of three implementations based on heuristics:
   *  - Thin: very efficient when D is small and/or N is large
   *  - Medium: other cases
   */
  const IdxType numSMs = raft::getMultiProcessorCount();
  if (D <= IdxType(256) || N >= IdxType(4) * numSMs) {
    coalescedReductionThinDispatcher(mgrp_dots,
                                     mgrp_data,
                                     n_groups,
                                     D,
                                     N,
                                     dev_d_ptr,
                                     dev_n_ptr,
                                     dots_offsets,
                                     data_offsets,
                                     init,
                                     stream,
                                     inplace,
                                     main_op,
                                     reduce_op,
                                     final_op);
  } else {
    coalescedReductionMediumDispatcher(mgrp_dots,
                                       mgrp_data,
                                       n_groups,
                                       D,
                                       N,
                                       dev_d_ptr,
                                       dev_n_ptr,
                                       dots_offsets,
                                       data_offsets,
                                       init,
                                       stream,
                                       inplace,
                                       main_op,
                                       reduce_op,
                                       final_op);
  }
}

}  // namespace Reduce
}  // namespace VertexDeg
}  // namespace Multigroups
}  // namespace Dbscan
}  // namespace ML
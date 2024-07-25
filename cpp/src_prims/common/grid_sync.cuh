/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <raft/util/cuda_utils.cuh>

namespace MLCommon {

/**
 * @brief The kind of synchronization that is needed
 */
enum SyncType {
  /** sync across all the blocks */
  ACROSS_ALL = 0,
  /** sync across all the locks along a given blockIdx.y and blockIdx.z */
  ACROSS_X
  ///@todo: ACROSS_Y, ACROSS_Z
};

/**
 * @brief A device-side structure to provide inter-block synchronization
 *
 * @note This does NOT provide synchronization across any arbitrary group of
 * threadblocks! Make sure you have read the documentation of SyncType enum to
 * know the list of supported synchronization 'modes'.
 *
 * @code{.cu}
 * CUML_KERNEL void kernel(void* workspace, SyncType type, ...) {
 *   GridSync gs(workspace, type);
 *   // do pre-sync work here
 *   // ...
 *   gs.sync();
 *   // do post-sync work here
 *   // ...
 * }
 *
 * SyncType type = ACROSS_ALL; // full grid-wide sync
 * char* workspace;
 * // allocate the workspace by getting to know the right size needed
 * size_t workspaceSize = GridSync::computeWorkspaceSize(gridDim, type);
 * RAFT_CUDA_TRY(cudaMalloc((void**)&workspace, workspaceSize);
 * // before the first usage of this workspace, initialize this to 0
 * // this is a one-time thing and if you're passing the same workspace
 * // to the same GridSync object inside the kernel and this workspace is
 * // exclusive, then subsequent memset calls are not needed
 * RAFT_CUDA_TRY(cudaMemset(workspace, 0, workspaceSize));
 * kernel<<<gridDim, blockDim>>>(workspace, type, ...);
 * RAFT_CUDA_TRY(cudaFree(workspace));
 * @endcode
 *
 * @note In order to call `GridSync::sync` method consecutively on the same
 * object inside the same kernel, make sure you set the 'multiSync' flag that is
 * passed `GridSync::computeWorkspaceSize` as well as `GridSync` constructor.
 * Having this flag not set, but trying to call `sync` method consecutively in
 * the same kernel using that same object can lead to deadlock and thus such a
 * usage is discouraged. Example follows:
 *
 * @code{.cu}
 * CUML_KERNEL void kernelMultiple(void* workspace, SyncType type, ...) {
 *   GridSync gs(workspace, type, true);
 *   ////// Part1 //////
 *   // do pre-sync work here
 *   // ...
 *   gs.sync();
 *   // do post-sync work here
 *   // ...
 *   ////// Part2 //////
 *   // do pre-sync work here
 *   // ...
 *   gs.sync();
 *   // do post-sync work here
 *   // ...
 *   ////// Part3 //////
 *   // do pre-sync work here
 *   // ...
 *   gs.sync();
 *   // do post-sync work here
 *   // ...
 * }
 * @endcode
 *
 * @todo Implement the lock-free synchronization approach described in this
 * paper: https://synergy.cs.vt.edu/pubs/papers/xiao-ipdps2010-gpusync.pdf
 * Probably cleaner to implement this as a separate class?
 */
struct GridSync {
  /**
   * @brief ctor
   * @param _workspace workspace needed for providing synchronization
   * @param _type synchronization type
   * @param _multiSync whether we need this object to perform multiple
   *  synchronizations in the same kernel call
   *
   * @note
   * <ol>
   * <li>All threads across all threadblocks must instantiate this object!
   * <li>
   *   Also, make sure that the workspace has been initialized to zero before
   *   the first usage of this workspace
   * </li>
   * <li>This workspace must not be used elsewhere concurrently</li>
   * </ol>
   */
  DI GridSync(void* _workspace, SyncType _type, bool _multiSync = false)
    : workspace((int*)_workspace), syncType(_type), multiSync(_multiSync)
  {
    if (syncType == ACROSS_X) {
      offset            = blockIdx.y + blockIdx.z * gridDim.y;
      stride            = gridDim.y * gridDim.z;
      int nBlksToArrive = gridDim.x;
      updateValue       = blockIdx.x == 0 ? -(nBlksToArrive - 1) : 1;
    } else {
      offset            = 0;
      stride            = 1;
      int nBlksToArrive = gridDim.x * gridDim.y * gridDim.z;
      updateValue =
        blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 ? -(nBlksToArrive - 1) : 1;
    }
  }

  /**
   * @brief Perform the synchronization
   *
   * @note All threads of all threadblocks must call this unconditionally!
   * There's no need to wrap this call between __syncthreads. That is taken
   * care of internally.
   */
  DI void sync()
  {
    int* arrivalTracker = workspace + offset;
    markArrived(arrivalTracker);
    waitForOthers((volatile int*)arrivalTracker);
    if (multiSync) { offset = offset < stride ? offset + stride : offset - stride; }
  }

  /**
   * @brief Computes workspace needed (in B) for the grid-sync
   * @param gridDim grid dimensions for the kernel to be launched
   * @param type synchronization type (this must the same as will be passed
   * eventually inside the kernel, while creating a device object of this
   * class)
   * @param multiSync whether we need this object to perform multiple
   *  synchronizations in the same kernel call
   */
  static size_t computeWorkspaceSize(const dim3& gridDim, SyncType type, bool multiSync = false)
  {
    int nblks   = type == ACROSS_X ? gridDim.y * gridDim.z : 1;
    size_t size = sizeof(int) * nblks;
    if (multiSync) { size *= 2; }
    return size;
  }

 private:
  /** workspace buffer */
  int* workspace;
  /** synchronization type */
  SyncType syncType;
  /** whether we need to perform multiple syncs in the same kernel call */
  bool multiSync;
  /** update value to be atomically updated by each arriving block */
  int updateValue;
  /** stride between 2 half of the workspace to ping-pong between */
  int stride;
  /** offset for the set of threadblocks in the current workspace */
  int offset;

  /**
   * @brief Register your threadblock to have arrived at the sync point
   * @param arrivalTracker the location that'll be atomically updated by all
   *  arriving threadblocks
   *
   * @note All threads of this threadblock must call this unconditionally!
   */
  DI void markArrived(int* arrivalTracker)
  {
    __syncthreads();
    if (masterThread()) {
      __threadfence();
      raft::myAtomicAdd(arrivalTracker, updateValue);
      __threadfence();
    }
  }

  /**
   * @brief Perform a wait until all the required threadblocks have arrived
   * at the sync point by calling the 'arrived' method.
   * @param gmemArrivedBlks the location that'd have been atomically updated
   *  by all arriving threadblocks
   *
   * @note All threads of all threadblocks must call this unconditionally!
   */
  DI void waitForOthers(volatile int* gmemArrivedBlks)
  {
    if (masterThread()) {
      int arrivedBlks = -1;
      do {
        arrivedBlks = *gmemArrivedBlks;
      } while (arrivedBlks != 0);
      __threadfence();
    }
    __syncthreads();
  }

  DI bool masterThread() const { return threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0; }
};  // struct GridSync

/**
 * @brief Helper method to have a group of threadblocks signal completion to
 *        others and also determine who's the last to arrive at this sync point
 * @param done_count location in global mem used to mark signal done of the
 *                   current threadblock.
 * @param nBlks number of blocks involved with this done-handshake
 * @param master which block is supposed to be considered as master in this
 *               process of handshake.
 * @param amIlast shared mem used for 'am i last' signal propagation to all the
 *                threads in the block
 * @return true if the current threadblock is the last to arrive else false
 *
 * @note This function should be entered by all threads in the block together.
 *       It is the responsibility of the calling code to ensure that before
 *       entering this function, all threads in this block really have completed
 *       whatever their individual tasks were.
 */
DI bool signalDone(int* done_count, int nBlks, bool master, int* amIlast)
{
  if (threadIdx.x == 0) {
    auto delta = master ? nBlks - 1 : -1;
    auto old   = atomicAdd(done_count, delta);
    *amIlast   = ((old + delta) == 0);
  }
  __syncthreads();
  return *amIlast;
}

};  // end namespace MLCommon

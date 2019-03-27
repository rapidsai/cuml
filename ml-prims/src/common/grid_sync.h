/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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


namespace MLCommon {

/**
 * @brief The kind of synchronization that is needed
 */
enum SyncType {
  /** sync across all the blocks */
  ACROSS_ALL= 0,
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
 */
struct GridSync {
    /**
     * @brief ctor
     * @param workspace workspace needed for providing synchronization
     * @param _type synchronization type
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
    DI GridSync(void* workspace, SyncType _type): syncType(_type) {
        int offset;
        if(syncType == ACROSS_X) {
            offset = blockIdx.y + blockIdx.z * gridDim.y;
            int nBlksToArrive = gridDim.x;
            updateValue = blockIdx.x == 0? -(nBlksToArrive - 1) : 1;
        } else {
            offset = 0;
            int nBlksToArrive = gridDim.x * gridDim.y * gridDim.z;
            updateValue = blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0?
                -(nBlksToArrive - 1) : 1;
        }
        arrivalTracker = ((int*)workspace) + offset;
    }

    /**
     * @brief Perform the synchronization
     *
     * @note All threads of all threadblocks must call this unconditionally!
     * There's no need to wrap this call between __syncthreads. That is taken
     * care of internally.
     */
    DI void sync() {
        markArrived();
        waitForOthers();
    }

    /**
     * @brief Computes workspace needed (in B) for the grid-sync
     * @param gridDim grid dimensions for the kernel to be launched
     * @param type synchronization type (this must the same as will be passed
     * eventually inside the kernel, while creating a device object of this
     * class)
     */
    static size_t computeWorkspaceSize(const dim3& gridDim, SyncType type) {
        int nblks = type == ACROSS_X? gridDim.y * gridDim.z : 1;
        return nblks * sizeof(int);
    }

private:
    /** synchronization type */
    SyncType syncType;
    /** arrival count monitor for the current group of blocks */
    int* arrivalTracker;
    /** update value to be atomically updated by each arriving block */
    int updateValue;


    /**
     * @brief Register your threadblock to have arrived at the sync point
     *
     * @note All threads of this threadblock must call this unconditionally!
     */
    DI void markArrived() {
        __syncthreads();
        if(threadIdx.x == 0) {
            __threadfence();
            atomicAdd(arrivalTracker, updateValue);
            __threadfence();
        }
    }

    /**
     * @brief Perform a wait until all the required threadblocks have arrived
     * at the sync point by calling the 'arrived' method.
     *
     * @note All threads of all threadblocks must call this unconditionally!
     */
    DI void waitForOthers() {
        if(threadIdx.x == 0) {
            int arrivedBlks = -1;
            volatile int* gmemArrivedBlks = (volatile int*)arrivalTracker;
            do {
                arrivedBlks = *gmemArrivedBlks;
            } while(arrivedBlks != 0);
            __threadfence();
        }
        __syncthreads();
    }
};

}; // end namespace MLCommon

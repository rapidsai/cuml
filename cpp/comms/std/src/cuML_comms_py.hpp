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

#include <nccl.h>

#include <cuML.hpp>

namespace ML {

bool ucx_enabled();

/**
   * @brief This function wraps the inject comms functions in
   * cpp/comms/std/include/cuML_comms.hpp to decouple the Python
   * layer from the optional UCX dependency in the C++ build. This
   * allows the Cython to compile without having to propagate the `WITH_UCX`
   * directive to that layer.
   * @param handle the cuml handle to inject a new communicator instance into
   * @param comm initialized nccl communicator
   * @param ucp_worker: ucp_worker_h instance for the current initialized ucp context
   * @param eps an array of ucp_ep_h endpoints to the other ucp workers in the cluster
   * @param size the size of the cluster (number of elements in eps)
   * @param rank rank of the current worker
   */
void inject_comms_py(cumlHandle *handle, ncclComm_t comm,

#ifdef WITH_UCX
                     void *ucp_worker, void *eps,
#else
                     void *, void *,
#endif
                     int size, int rank);

/**
   * @brief This function follows the design of the wrapper function in
   * cpp/comms/std/include/cuML_comms.hpp to decouple the Python layer
   * injection functions from the C++ layer functions.
   * @param handle the cuml handle to inject a new communicator instance into
   * @param comm initialized nccl communicator
   * @param size the size of the cluster (number of elements in eps)
   * @param rank rank of the current worker
   */

void inject_comms_py_coll(cumlHandle *handle, ncclComm_t comm, int size,
                          int rank);

/**
   * @brief Stores the given character array on the given ncclUniqueId struct.
   * @param id the ncclUniqueId struct instance to store the given character array
   * @param uniqueId the unique id char array to store on the ncclUniqueId
   */
void ncclUniqueIdFromChar(ncclUniqueId *id, char *uniqueId, int size);

/**
   * @brief Returns a new ncclUniqueId from the nccl API and stores in
   * the given character array for serialization
   */
void get_unique_id(char *uid, int size);
}  // namespace ML

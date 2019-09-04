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

#ifdef WITH_UCX
#include <ucp/api/ucp.h>
#endif

#include <cuML.hpp>

namespace ML {

/**
 * @brief Given initialized comms handles for NCCL and UCP, this function builds a
 * cumlCommunicator object and injects it into the given cumlHandle instance.
 * @param handle the cuml handle to inject a new communicator instance into
 * @param comm initialized nccl communicator
 * @param ucp_worker the ucp_worker for the current initialized ucp context
 * @param eps an array of endpoints to the other ucp workers in the cluster
 * @param size the size of the cluster (number of elements in eps)
 * @param rank rank of the current worker
 */
#ifdef WITH_UCX
void inject_comms(cumlHandle &handle, ncclComm_t comm, ucp_worker_h ucp_worker,
                  ucp_ep_h *eps, int size, int rank);
#endif

/**
 * @brief Given an initialized comms handle for NCCL, this function builds a
 * cumlCommunicator object and injects it into the given cumlHandle instance.
 * The underlying cumlCommunicator will only have support for collective
 * communications functions.
 * @param handle the cuml handle to inject a new communicator instance into
 * @param comm initialized nccl communicator
 * @param size the size of the cluster
 * @param rank rank of the current worker
 */
void inject_comms(cumlHandle &handle, ncclComm_t comm, int size, int rank);

}  // end namespace ML

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
#include <ucp/api/ucp.h>

#include <cuML.hpp>

namespace ML {

/**
 * Given initialized comms handles for NCCL and UCP, this function builds a cumlCommunicator
 * object and injects it into the given cumlHandle instance.
 */
void inject_comms(cumlHandle& handle, ncclComm_t comm, ucp_worker_h *ucp_worker, ucp_ep_h **eps, int size, int rank);

} // end namespace ML

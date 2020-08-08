/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include "cuML_comms_py.hpp"

namespace ML {

void inject_comms_py(cumlHandle *handle, ncclComm_t comm, void *ucp_worker,
                     void *eps, int size, int rank) {
    auto eps_sp = std::make_shared<ucp_ep_h *>(new ucp_ep_h[size]);

    auto size_t_ep_arr = reinterpret_cast<size_t *>(eps);

    for (int i = 0; i < size; i++) {
    size_t ptr = size_t_ep_arr[i];
    auto ucp_ep_v = reinterpret_cast<ucp_ep_h *>(*eps_sp);

    if (ptr != 0) {
        auto eps_ptr = reinterpret_cast<ucp_ep_h>(size_t_ep_arr[i]);
        ucp_ep_v[i] = eps_ptr;
    } else {
        ucp_ep_v[i] = nullptr;
    }
    }

    std::shared_ptr<raft::mr::device::allocator> d_alloc = std::make_shared<raft::mr::device::default_allocator>();
    cudaStream_t stream = handle->getStream();

    auto communicator = std::make_shared<raft::comms::comms_t>(std::unique_ptr<raft::comms::comms_iface>(
        new raft::comms::std_comms(comm, (ucp_worker_h)ucp_worker, eps_sp,
                                   size, rank, d_alloc, stream)));

    handle->getImpl().setCommunicator(communicator);
}

void inject_comms_py_coll(cumlHandle *handle, ncclComm_t comm, int size,
                          int rank) {

    std::shared_ptr<raft::mr::device::allocator> d_alloc = std::make_shared<raft::mr::device::default_allocator>();
    cudaStream_t stream = handle->getStream();

    auto communicator = std::make_shared<raft::comms::comms_t>(std::unique_ptr<raft::comms::comms_iface>(
        new raft::comms::std_comms(comm, size, rank, d_alloc, stream)));

    handle->getImpl().setCommunicator(communicator);
}

} // namespace ML
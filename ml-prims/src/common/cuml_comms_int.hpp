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

#include <memory>

namespace MLCommon {

class cumlCommunicator_iface;

/**
 * Communicator class intended to be used by cuML and ml-prims.
 *
 * cumlCommunicator is part of the public interface of cuML-comms, but
 * not intended to be used outside of cuML and ml-prims. The public
 * interface of cuML-comms for cuML users is in cuML_comms.hpp.
 * cumlCommunicator should not have any external dependencies to e.g.
 * mpi.h or nccl.h. This dependency is hidden in cumlCommunicator_impl.
 * A cuML user needs to build cuML-comms for the MPI or NCCL version he
 * is using. This enables cumlCommunicator to be used in 
 * closed source components like multi GPU ml-prims without a direct
 * dependency to the MPI or NCCL version a cuML user has build cuML-comms
 * for.
 */
class cumlCommunicator {
public:
    typedef unsigned int request_t;
    cumlCommunicator() =delete;
    cumlCommunicator(std::unique_ptr<cumlCommunicator_iface> impl);

    int getSize() const;
    int getRank() const;

    void barrier() const;

    void isend(const void *buf, std::size_t size, int dest, int tag, request_t *request) const;
    void irecv(void *buf, std::size_t size, int source, int tag, request_t *request) const;

    template<typename T>
    void isend(const T *buf, int n, int dest, int tag, request_t *request) const
    {
        isend(static_cast<const void*>(buf), n*sizeof(T), dest, tag, request);
    }

    template<typename T>
    void irecv(T *buf, int n, int source, int tag, request_t *request) const
    {
        irecv(static_cast<void*>(buf), n*sizeof(T), source, tag, request);
    }

    void waitall(int count, request_t array_of_requests[]) const;

private:
    std::unique_ptr<cumlCommunicator_iface> _impl;
};

} // end namespace MLCommon

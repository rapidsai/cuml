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

#include "cuml_comms_int.hpp"

namespace MLCommon {

/**
 * cuML communicator plugin interface. This is part of ml-prims because
 * multi GPU ml-prims need it and we want to avoid a dependency of ml-prims to
 * cuML. This is not part of cuML-comms as we would have a circular dependency
 * of cuML-comms and cuML/ml-prims (cuML-comms to cuML/ml-prims due to the 
 * implementation of initialize_mpi_comms in cuML-comms, which depdends on 
 * cumlHandle and cuML/ml-prims to cuML-comms to cumlCommunicator_iface if
 * cumlCommunicator_iface would be part of cuML-comms).
 */
class cumlCommunicator_iface {
public:
    typedef cumlCommunicator::request_t     request_t;
    typedef cumlCommunicator::datatype_t    datatype_t;
    typedef cumlCommunicator::op_t          op_t;

    virtual ~cumlCommunicator_iface();

    virtual int getSize() const =0;
    virtual int getRank() const =0;

    virtual void barrier() const =0;

    virtual void isend(const void *buf, int size, int dest, int tag, request_t *request) const =0;

    virtual void irecv(void *buf, int size, int source, int tag, request_t *request) const =0;

    virtual void waitall(int count, request_t array_of_requests[]) const =0;

    virtual void allreduce(const void* sendbuff, void* recvbuff, int count, datatype_t datatype, op_t op, cudaStream_t stream) const =0;

    virtual void bcast(void* buff, int count, datatype_t datatype, int root, cudaStream_t stream) const =0;

    virtual void reduce(const void* sendbuff, void* recvbuff, int count, datatype_t datatype, op_t op, int root, cudaStream_t stream) const =0;

    virtual void allgather(const void* sendbuff, void* recvbuff, int sendcount, datatype_t datatype, cudaStream_t stream) const =0;

    virtual void allgatherv(const void *sendbuf, void *recvbuf, const int recvcounts[], const int displs[], datatype_t datatype, cudaStream_t stream) const =0;

    virtual void reducescatter(const void* sendbuff, void* recvbuff, int recvcount, datatype_t datatype, op_t op, cudaStream_t stream) const =0;
};

} // end namespace ML

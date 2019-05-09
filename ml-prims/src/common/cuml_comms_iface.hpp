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

class cumlCommunicator_iface {
public:
    typedef cumlCommunicator::request_t     request_t;
    typedef cumlCommunicator::datatype_t    datatype_t;
    typedef cumlCommunicator::op_t          op_t;

    virtual ~cumlCommunicator_iface();

    virtual int getSize() const =0;
    virtual int getRank() const =0;

    virtual void barrier() const =0;

    virtual void isend(const void *buf, std::size_t size, int dest, int tag, request_t *request) const =0;

    virtual void irecv(void *buf, std::size_t size, int source, int tag, request_t *request) const =0;

    virtual void waitall(int count, request_t array_of_requests[]) const =0;

    virtual void allreduce(const void* sendbuff, void* recvbuff, size_t count, datatype_t datatype, op_t op, cudaStream_t stream) const =0;
};

} // end namespace ML

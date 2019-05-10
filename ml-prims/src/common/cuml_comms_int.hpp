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

#include <cuda_runtime.h>

namespace MLCommon {

class cumlCommunicator_iface;

/**
 * Communicator class intended to be used by cuML and ml-prims.
 *
 * cumlCommunicator needs an implementation of cumlCommunicator_iface.
 * The propsal is that this comes from a seperate library (cuML-comms).
 * This enables a cuML user to build cuML-comms for the comms stack version he
 * is using. The rational for this choice is that cumlCommunicator can be used
 * in closed source components like multi GPU ml-prims without a direct
 * dependency to the users comms stack.
 *
 * The methods exposed by cumlCommunicator are thin wrappers around NCCL and
 * a comms stack with MPI semantics.
 */
class cumlCommunicator {
public:
    typedef unsigned int request_t;
    enum datatype_t { CHAR, UINT8, INT, UINT, INT64, UINT64, FLOAT, DOUBLE };
    enum op_t { SUM, PROD, MIN, MAX };

    template<typename T>
    datatype_t getDataType() const;

    cumlCommunicator() =delete;
    cumlCommunicator(std::unique_ptr<cumlCommunicator_iface> impl);

    /**
     * Returns the size of the group associated with the underlying communicator.
     */
    int getSize() const;
    /**
     * Determines the rank of the calling process in the underlying communicator.
     */
    int getRank() const;

    /**
     * Synchronization all ranks for the underlying communicator.
     */
    void barrier() const;

    /**
     * Starts a nonblocking send following the semantics of MPI_Isend
     *
     * @param[in]   buf     address of send buffer (can be a CPU or GPU pointer)
     * @param[in]   n       size of the message to send in bytes
     * @param[in]   dest    rank of destination
     * @param[in]   tag     message tag
     * @param[out]  request communication request (handle)
     */
    void isend(const void *buf, std::size_t size, int dest, int tag, request_t *request) const;
    /**
     * Starts a nonblocking receive following the semantics of MPI_Irecv
     *
     * @param[in]   buf     address of receive buffer (can be a CPU or GPU pointer)
     * @param[in]   n       size of the message to receive in bytes
     * @param[in]   source  rank of source
     * @param[in]   tag     message tag
     * @param[out]  request communication request (handle)
     */
    void irecv(void *buf, std::size_t size, int source, int tag, request_t *request) const;

    /**
     * Convience wrapper around isend taking a void buffer as input.
     *
     * @param[in]   buf     address of send buffer (can be a CPU or GPU pointer)
     * @param[in]   n       number of elements to send
     * @param[in]   dest    rank of destination
     * @param[in]   tag     message tag
     * @param[out]  request communication request (handle)
     */
    template<typename T>
    void isend(const T *buf, int n, int dest, int tag, request_t *request) const
    {
        isend(static_cast<const void*>(buf), n*sizeof(T), dest, tag, request);
    }

    /**
     * Convience wrapper around irecv taking a void buffer as input.
     *
     * @param[in]   buf     address of receive buffer (can be a CPU or GPU pointer)
     * @param[in]   n       number of elements to receive
     * @param[in]   source  rank of source
     * @param[in]   tag     message tag
     * @param[out]  request communication request (handle)
     */
    template<typename T>
    void irecv(T *buf, int n, int source, int tag, request_t *request) const
    {
        irecv(static_cast<void*>(buf), n*sizeof(T), source, tag, request);
    }

    /**
     * Waits for all given communication requests to complete following the semantics of MPI_Waitall.
     *
     * @param[in]   count               number of requests
     * @param[in]   array_of_requests   array of request handles
     */
    void waitall(int count, request_t array_of_requests[]) const;

    /**
     * Reduce data arrays of length count in sendbuff using op operation and leaves identical copies of the 
     * result on each recvbuff.
     *
     * Follows the semantics of ncclAllReduce. In-place operation will happen if sendbuff == recvbuff .
     *
     * @param[in]   sendbuff    address of GPU accessible send buffer
     * @param[in]   recvbuff    address of GPU accessible receive buffer (might alias with sendbuff)
     * @param[in]   count       number of elements in sendbuff and recvbuff
     * @param[in]   op          reduction operation to perform.
     * @param[in]   stream      stream to submit this asynchronous (with respect to the CPU) operation to
     */
    void allreduce(const void* sendbuff, void* recvbuff, size_t count, datatype_t datatype, op_t op, cudaStream_t stream) const;

    /**
     * Convience wrapper around allreduce taking a void buffers as input deducing datatype_t from T.
     *
     * @param[in]   sendbuff    address of GPU accessible send buffer
     * @param[in]   recvbuff    address of GPU accessible receive buffer (might alias with sendbuff)
     * @param[in]   count       number of elements in sendbuff and recvbuff
     * @param[in]   op          reduction operation to perform.
     * @param[in]   stream      stream to submit this asynchronous (with respect to the CPU) operation to
     */
    template<typename T>
    void allreduce(const T* sendbuff, T* recvbuff, size_t count, op_t op, cudaStream_t stream) const
    {
        allreduce(sendbuff, recvbuff, count, getDataType<T>(), op, stream);
    }

private:
    std::unique_ptr<cumlCommunicator_iface> _impl;
};

} // end namespace MLCommon

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

  enum status_t { commStatusSuccess, commStatusError, commStatusAbort };

  template <typename T>
  datatype_t getDataType() const;

  cumlCommunicator() = delete;
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
     * Creates new communicators based on colors and keys following the sematics of MPI_Comm_split.
     *
     * Note: Issuing concurrent communication requests to overlapping communicators can cause a 
     *       deadlock.
     *
     * @param[in]   color   Control of subset assignment (nonnegative integer)
     * @param[in]   key     Control of rank assignment
     * @return              new communicator instance containing only the ranks with the same color
     */
  cumlCommunicator commSplit(int color, int key) const;

  /**
     * Synchronization all ranks for the underlying communicator.
     */
  void barrier() const;

  status_t syncStream(cudaStream_t stream) const;

  /**
     * Starts a nonblocking send following the semantics of MPI_Isend
     *
     * @param[in]   buf     address of send buffer (can be a CPU or GPU pointer)
     * @param[in]   n       size of the message to send in bytes
     * @param[in]   dest    rank of destination
     * @param[in]   tag     message tag
     * @param[out]  request communication request (handle)
     */
  void isend(const void* buf, int size, int dest, int tag,
             request_t* request) const;
  /**
     * Starts a nonblocking receive following the semantics of MPI_Irecv
     *
     * @param[in]   buf     address of receive buffer (can be a CPU or GPU pointer)
     * @param[in]   n       size of the message to receive in bytes
     * @param[in]   source  rank of source
     * @param[in]   tag     message tag
     * @param[out]  request communication request (handle)
     */
  void irecv(void* buf, int size, int source, int tag,
             request_t* request) const;

  /**
     * Convience wrapper around isend deducing message size from sizeof(T).
     *
     * @param[in]   buf     address of send buffer (can be a CPU or GPU pointer)
     * @param[in]   n       number of elements to send
     * @param[in]   dest    rank of destination
     * @param[in]   tag     message tag
     * @param[out]  request communication request (handle)
     */
  template <typename T>
  void isend(const T* buf, int n, int dest, int tag, request_t* request) const {
    isend(static_cast<const void*>(buf), n * sizeof(T), dest, tag, request);
  }

  /**
     * Convience wrapper around irecv deducing message size from sizeof(T).
     *
     * @param[in]   buf     address of receive buffer (can be a CPU or GPU pointer)
     * @param[in]   n       number of elements to receive
     * @param[in]   source  rank of source
     * @param[in]   tag     message tag
     * @param[out]  request communication request (handle)
     */
  template <typename T>
  void irecv(T* buf, int n, int source, int tag, request_t* request) const {
    irecv(static_cast<void*>(buf), n * sizeof(T), source, tag, request);
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
     * @param[in]   datatype    data type of sendbuff and recvbuff
     * @param[in]   op          reduction operation to perform.
     * @param[in]   stream      stream to submit this asynchronous (with respect to the CPU) operation to
     */
  void allreduce(const void* sendbuff, void* recvbuff, int count,
                 datatype_t datatype, op_t op, cudaStream_t stream) const;

  /**
     * Convience wrapper around allreduce deducing datatype_t from T.
     */
  template <typename T>
  void allreduce(const T* sendbuff, T* recvbuff, int count, op_t op,
                 cudaStream_t stream) const {
    allreduce(sendbuff, recvbuff, count, getDataType<T>(), op, stream);
  }

  /**
     * Copies count elements from buff on the root rank to all ranks buff.
     *
     * Follows the semantics of ncclBcast.
     *
     * @param[in]   buff        address of GPU accessible buffer
     * @param[in]   count       number of elements in buff
     * @param[in]   datatype    data type of buff
     * @param[in]   root        rank of broadcast root
     * @param[in]   stream      stream to submit this asynchronous (with respect to the CPU) operation to
     */
  void bcast(void* buff, int count, datatype_t datatype, int root,
             cudaStream_t stream) const;

  /**
     * Convience wrapper around bcast deducing datatype_t from T.
     */
  template <typename T>
  void bcast(T* buff, int count, int root, cudaStream_t stream) const {
    bcast(buff, count, getDataType<T>(), root, stream);
  }

  /**
     * Reduce data arrays of length count in sendbuff into recvbuff on the root rank using the op operation.
     * recvbuff is only used on rank root and ignored for other ranks. 
     *
     * Follows the semantics of ncclReduce. In-place operation will happen if sendbuff == recvbuff .
     *
     * @param[in]   sendbuff    address of GPU accessible send buffer
     * @param[in]   recvbuff    address of GPU accessible receive buffer (might alias with sendbuff)
     * @param[in]   count       number of elements in sendbuff and recvbuff
     * @param[in]   datatype    data type of sendbuff and recvbuff
     * @param[in]   op          reduction operation to perform.
     * @param[in]   root        rank of broadcast root
     * @param[in]   stream      stream to submit this asynchronous (with respect to the CPU) operation to
     */
  void reduce(const void* sendbuff, void* recvbuff, int count,
              datatype_t datatype, op_t op, int root,
              cudaStream_t stream) const;

  /**
     * Convience wrapper around reduce deducing datatype_t from T.
     */
  template <typename T>
  void reduce(const T* sendbuff, T* recvbuff, int count, op_t op, int root,
              cudaStream_t stream) const {
    reduce(sendbuff, recvbuff, count, getDataType<T>(), op, root, stream);
  }

  /**
     * Gather sendcount values from all GPUs into recvbuff, receiving data from rank i at offset i*sendcount.
     *
     * Note : This assumes the receive count is equal to nranks*sendcount, which means that recvbuff should
     * have a size of at least nranks*sendcount elements.
     *
     * In-place operation will happen if sendbuff == recvbuff + rank * sendcount.
     *
     * Follows the semantics of ncclAllGather.
     *
     * @param[in]   sendbuff    address of GPU accessible send buffer
     * @param[in]   recvbuff    address of GPU accessible receive buffer (might alias with sendbuff)
     * @param[in]   sendcount   number of elements in sendbuff and recvbuff
     * @param[in]   datatype    data type of sendbuff and recvbuff
     * @param[in]   stream      stream to submit this asynchronous (with respect to the CPU) operation to
     */
  void allgather(const void* sendbuff, void* recvbuff, int sendcount,
                 datatype_t datatype, cudaStream_t stream) const;

  /**
     * Convience wrapper around allgather deducing datatype_t from T.
     */
  template <typename T>
  void allgather(const T* sendbuff, T* recvbuff, int sendcount,
                 cudaStream_t stream) const {
    allgather(sendbuff, recvbuff, sendcount, getDataType<T>(), stream);
  }

  /**
     * Gathers data from all processes and delivers it to all. Each process may contribute a
     * different amount of data.
     *
     * Semantics are equivalent to:
     *
     *    for (int root = 0; root < getSize(); ++root) {
     *        ncclBroadcast(sendbuf,
     *                      static_cast<char*>(recvbuf)+displs[root]*sizeof(datatype), recvcounts[root],
     *                      datatype, root, nccl_comm, stream);
     *    }
     *
     * @param[in]   sendbuff    address of GPU accessible send buffer
     * @param[in]   recvbuff    address of GPU accessible receive buffer (might alias with sendbuff)
     * @param[in]   recvcounts  array (of length group size) containing the number of elements that are
     *                          received from each process.
     * @param[in]   displs      array (of length group size). Entry i specifies the displacement
     *                          (relative to recvbuf) at which to place the incoming data from process i.
     * @param[in]   datatype    data type of sendbuff and recvbuff
     * @param[in]   stream      stream to submit this asynchronous (with respect to the CPU) operation to
     */
  void allgatherv(const void* sendbuf, void* recvbuf, const int recvcounts[],
                  const int displs[], datatype_t datatype,
                  cudaStream_t stream) const;

  /**
     * Convience wrapper around allgatherv deducing datatype_t from T.
     */
  template <typename T>
  void allgatherv(const void* sendbuf, void* recvbuf, const int recvcounts[],
                  const int displs[], cudaStream_t stream) const {
    allgatherv(sendbuf, recvbuf, recvcounts, displs, getDataType<T>(), stream);
  }

  /**
     * Reduce data in sendbuff from all GPUs using the op operation and leave the reduced result scattered
     * over the devices so that the recvbuff on rank i will contain the i-th block of the result.
     *
     * Note: This assumes the send count is equal to nranks*recvcount, which means that sendbuff should have
     * a size of at least nranks*recvcount elements.
     *
     * Follows the semantics of ncclReduceScatter. 
     *
     * @param[in]   sendbuff    address of GPU accessible send buffer
     * @param[in]   recvbuff    address of GPU accessible receive buffer
     * @param[in]   recvcount   number of elements to receive to recvbuff
     * @param[in]   datatype    data type of sendbuff and recvbuff
     * @param[in]   op          reduction operation to perform.
     * @param[in]   stream      stream to submit this asynchronous (with respect to the CPU) operation to
     */
  void reducescatter(const void* sendbuff, void* recvbuff, int recvcount,
                     datatype_t datatype, op_t op, cudaStream_t stream) const;

  /**
     * Convience wrapper around reducescatter deducing datatype_t from T.
     */
  template <typename T>
  void reducescatter(const void* sendbuff, void* recvbuff, int recvcount,
                     datatype_t datatype, op_t op, cudaStream_t stream) const {
    reducescatter(sendbuff, recvbuff, recvcount, getDataType<T>(), op, stream);
  }

 private:
  std::unique_ptr<cumlCommunicator_iface> _impl;
};

}  // end namespace MLCommon

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

#include <raft/comms/comms.hpp>

namespace MLCommon {


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
class cumlCommunicator : public raft::comms::comms_t {
 public:
  typedef unsigned int request_t;
  enum datatype_t { CHAR, UINT8, INT, UINT, INT64, UINT64, FLOAT, DOUBLE };
  enum op_t { SUM, PROD, MIN, MAX };

  static const int CUML_ANY_SOURCE = -1;

  /**
   * The resulting status of distributed stream synchronization
   */
  enum status_t {
    commStatusSuccess,  // Synchronization successful
    commStatusError,    // An error occured querying sync status
    commStatusAbort
  };  // A failure occured in sync, queued operations aborted

  template <typename T>
  datatype_t getDataType() const {
     return static_cast<datatype_t>(raft::comms::get_type<T>());
  }

  cumlCommunicator() = delete;
//   cumlCommunicator(std::unique_ptr<cumlCommunicator_iface> impl);

  /**
     * Returns the size of the group associated with the underlying communicator.
     */
  int getSize() const {
     return raft::comms::comms_t::get_size();
  }
  /**
     * Determines the rank of the calling process in the underlying communicator.
     */
  int getRank() const {
     return raft::comms::comms_t::get_rank();
  }

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
  cumlCommunicator commSplit(int color, int key) const {
      ASSERT(false,
      "ERROR: commSplit called but not yet supported in this comms "
      "implementation.");
  }

  /**
     * Synchronization of all ranks for the underlying communicator.
     */
  void barrier() const {
     raft::comms::comms_t::barrier();
  }

  /**
   * Synchronization of all ranks for the current stream. This allows different cumlCommunicator
   * implementations to provide custom handling of asynchronous errors, such as the failure of
   * ranks during collective communication operations.
   *
   * In the case where commStatusAbort is returned, the underlying comms implementation may need
   * to be re-initialized.
   *
   * A status of commStatusError should be thrown if an error occurs when querying the stream
   * sync status of the underlying communicator.
   *
   * @param[in] stream  the stream to synchronize
   * @return            resulting status of the synchronization.
   */
  status_t syncStream(cudaStream_t stream) const {
     return static_cast<status_t>(raft::comms::comms_t::sync_stream(stream));
  }

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
    raft::comms::comms_t::isend(buf, n, dest, tag, request);
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
    raft::comms::comms_t::irecv(buf, n, source, tag, request);
  }

  /**
     * Waits for all given communication requests to complete following the semantics of MPI_Waitall.
     *
     * @param[in]   count               number of requests
     * @param[in]   array_of_requests   array of request handles
     */
  void waitall(int count, request_t array_of_requests[]) const {
     raft::comms::comms_t::waitall(count, array_of_requests);
  }

  /**
     * Convience wrapper around allreduce deducing datatype_t from T.
     */
  template <typename T>
  void allreduce(const T* sendbuff, T* recvbuff, int count, op_t op,
                 cudaStream_t stream) const {
    raft::comms::comms_t::allreduce(sendbuff, recvbuff, count, static_cast<raft::comms::op_t>(op), stream);
  }

  /**
     * Convience wrapper around bcast deducing datatype_t from T.
     */
  template <typename T>
  void bcast(T* buff, int count, int root, cudaStream_t stream) const {
    raft::comms::comms_t::bcast(buff, count, root, stream);
  }

  /**
     * Convience wrapper around reduce deducing datatype_t from T.
     */
  template <typename T>
  void reduce(const T* sendbuff, T* recvbuff, int count, op_t op, int root,
              cudaStream_t stream) const {
    raft::comms::comms_t::reduce(sendbuff, recvbuff, count, static_cast<raft::comms::op_t>(op), root, stream);
  }

  /**
     * Convience wrapper around allgather deducing datatype_t from T.
     */
  template <typename T>
  void allgather(const T* sendbuff, T* recvbuff, int sendcount,
                 cudaStream_t stream) const {
    raft::comms::comms_t::allgather(sendbuff, recvbuff, sendcount, stream);
  }

  /**
     * Convience wrapper around allgatherv deducing datatype_t from T.
     */
  template <typename T>
  void allgatherv(const T* sendbuf, T* recvbuf, const int recvcounts[],
                  const int displs[], cudaStream_t stream) const {
    raft::comms::comms_t::allgatherv(sendbuf, static_cast<size_t*>(recvbuf), recvcounts, displs, stream);
  }

  /**
     * Convience wrapper around reducescatter deducing datatype_t from T.
     */
  template <typename T>
  void reducescatter(const T* sendbuff, T* recvbuff, int recvcount,
                     op_t op, cudaStream_t stream) const {
    raft::comms::comms_t::reducescatter(sendbuff, recvbuff, recvcount, static_cast<raft::comms::op_t>(op), stream);
  }

};

}  // end namespace MLCommon

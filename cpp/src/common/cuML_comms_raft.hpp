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

#include <common/cuml_comms_int.hpp>
#include <raft/comms/comms.hpp>

namespace MLCommon {

class cumlCommunicator_raft : public cumlCommunicator {

public:
    cumlCommunicator_raft() = delete;
    cumlCommunicator_raft(const raft::comms::comms_t& raftComms) {
        _raftComms = &raftComms;
    }
    ~cumlCommunicator_raft() {}

  int getSize() const {
      return _raftComms->get_size();
  }

  int getRank() const {
      return _raftComms->get_rank();
  }

  cumlCommunicator_raft commSplit(int color, int key) const {
    ASSERT(false,
        "ERROR: commSplit called but not yet supported in this comms "
        "implementation.");
  }

  void barrier() const {
      _raftComms->barrier();
  }

  status_t syncStream(cudaStream_t stream) const {
      return (MLCommon::cumlCommunicator::status_t) _raftComms->sync_stream(stream);
  }

  template <typename T>
  void isend(const T* buf, int n, int dest, int tag, request_t* request) const {
    _raftComms->isend(buf, n, dest, tag, request);
  }

  template <typename T>
  void irecv(T* buf, int n, int source, int tag, request_t* request) const {
    _raftComms->irecv(buf, n, source, tag, request);
  }

  void waitall(int count, request_t array_of_requests[]) const {
      _raftComms->waitall(count, array_of_requests);
  }

  template <typename T>
  void allreduce(const T* sendbuff, T* recvbuff, int count, op_t op,
                 cudaStream_t stream) const {
    _raftComms->allreduce(sendbuff, recvbuff, count, op, stream);
  }

  template <typename T>
  void bcast(T* buff, int count, int root, cudaStream_t stream) const {
    _raftComms->bcast(buff, count, root, stream);
  }

  template <typename T>
  void reduce(const T* sendbuff, T* recvbuff, int count, op_t op, int root,
              cudaStream_t stream) const {
    _raftComms->reduce(sendbuff, recvbuff, count, op, root, stream);
  }

  template <typename T>
  void allgather(const T* sendbuff, T* recvbuff, int sendcount,
                 cudaStream_t stream) const {
    _raftComms->allgather(sendbuff, recvbuff, sendcount, stream);
  }

  template <typename T>
  void allgatherv(const void* sendbuf, void* recvbuf, const int recvcounts[],
                  const int displs[], cudaStream_t stream) const {
    _raftComms->allgatherv(sendbuf, recvbuf, (size_t *) recvcounts, displs, stream);
  }

  template <typename T>
  void reducescatter(const T* sendbuff, T* recvbuff, int recvcount,
                     datatype_t datatype, op_t op, cudaStream_t stream) const {
    _raftComms->reducescatter(sendbuff, recvbuff, recvcount, op, stream);
  }

private:
    const raft::comms::comms_t* _raftComms;

};

} // end namespace MLCommon
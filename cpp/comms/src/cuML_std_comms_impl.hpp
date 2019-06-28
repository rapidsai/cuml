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

#include <unordered_map>
#include <unordered_set>
#include <utility>

#include <nccl.h>

#if WITH_UCX == 1
#include <ucp/api/ucp.h>
#endif

#include <common/cuml_comms_iface.hpp>

#if WITH_UCX == 1
struct ucx_context {
  int completed;
  bool needs_release = true;
};
#endif

namespace ML {

class cumlStdCommunicator_impl : public MLCommon::cumlCommunicator_iface {
 public:
  cumlStdCommunicator_impl() = delete;

#if WITH_UCX == 1
  cumlStdCommunicator_impl(ncclComm_t comm, ucp_worker_h ucp_worker,
                           std::shared_ptr<ucp_ep_h*> eps, int size, int rank);
#endif

  cumlStdCommunicator_impl(ncclComm_t comm, int size, int rank);

  virtual ~cumlStdCommunicator_impl();

  virtual int getSize() const;
  virtual int getRank() const;

  virtual std::unique_ptr<MLCommon::cumlCommunicator_iface> commSplit(
    int color, int key) const;

  virtual void barrier() const;

  virtual void isend(const void* buf, int size, int dest, int tag,
                     request_t* request) const;

  virtual void irecv(void* buf, int size, int source, int tag,
                     request_t* request) const;

  virtual void waitall(int count, request_t array_of_requests[]) const;

  virtual void allreduce(const void* sendbuff, void* recvbuff, int count,
                         datatype_t datatype, op_t op,
                         cudaStream_t stream) const;

  virtual void bcast(void* buff, int count, datatype_t datatype, int root,
                     cudaStream_t stream) const;

  virtual void reduce(const void* sendbuff, void* recvbuff, int count,
                      datatype_t datatype, op_t op, int root,
                      cudaStream_t stream) const;

  virtual void allgather(const void* sendbuff, void* recvbuff, int sendcount,
                         datatype_t datatype, cudaStream_t stream) const;

  virtual void allgatherv(const void* sendbuf, void* recvbuf,
                          const int recvcounts[], const int displs[],
                          datatype_t datatype, cudaStream_t stream) const;

  virtual void reducescatter(const void* sendbuff, void* recvbuff,
                             int recvcount, datatype_t datatype, op_t op,
                             cudaStream_t stream) const;

 private:
  ncclComm_t _nccl_comm;

#if WITH_UCX == 1
  ucp_worker_h _ucp_worker;
  std::shared_ptr<ucp_ep_h*> _ucp_eps;
  mutable request_t _next_request_id;
  mutable std::unordered_map<request_t, struct ucx_context*>
    _requests_in_flight;
  mutable std::unordered_set<request_t> _free_requests;
#endif

  int _size;
  int _rank;
};

}  // end namespace ML

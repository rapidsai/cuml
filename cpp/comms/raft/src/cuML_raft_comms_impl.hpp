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

#pragma once

#include <common/cuml_comms_iface.hpp>

#include <raft/comms/std_comms.hpp>

namespace ML {

/**
 * @brief A cumlCommunicator implementation capable of running collective communications
 * with NCCL and point-to-point-communications with UCX. Note that the latter is optional.
 *
 * Underlying comms, like NCCL and UCX, should be initialized and ready for use,
 * and maintained, outside of the cuML Comms lifecycle. This allows us to decouple the
 * ownership of the actual comms from cuml so that they can also be used outside of cuml.
 *
 * For instance, nccl-py can be used to bootstrap a ncclComm_t before it is
 * used to construct a cuml comms instance. UCX endpoints can be bootstrapped
 * in Python using ucx-py, before being used to construct a cuML comms instance.
 */
class cumlRAFTCommunicator_impl : public MLCommon::cumlCommunicator_iface {
 public:
  cumlRAFTCommunicator_impl() = delete;

  /**
   * @brief Constructor for collective + point-to-point operation.
   * @param comm initialized nccl comm
   * @param ucp_worker initialized ucp_worker instance
   * @param eps shared pointer to array of ucp endpoints
   * @param size size of the cluster
   * @param rank rank of the current worker
   */
  cumlRAFTCommunicator_impl(ncclComm_t comm, ucp_worker_h ucp_worker,
                           std::shared_ptr<ucp_ep_h*> eps, int size, int rank,
                           std::shared_ptr<raft::mr::device::allocator> device_allocator=nullptr, cudaStream_t stream=NULL);

  /**
   * @brief constructor for collective-only operation
   * @param comm initilized nccl communicator
   * @param size size of the cluster
   * @param rank rank of the current worker
   */
  cumlRAFTCommunicator_impl(ncclComm_t comm, int size, int rank, std::shared_ptr<raft::mr::device::allocator> device_allocator=nullptr, cudaStream_t stream=NULL);

  virtual ~cumlRAFTCommunicator_impl();

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

  virtual status_t syncStream(cudaStream_t stream) const;

  std::shared_ptr<raft::comms::comms_t> getRaftComms() const;

 private:
  std::shared_ptr<raft::comms::comms_t> _raftComms;

  void get_request_id(request_t* req) const;
};

inline cumlRAFTCommunicator_impl& cast_comms(MLCommon::cumlCommunicator_iface& comms) {
  return dynamic_cast<cumlRAFTCommunicator_impl&>(comms);
}

}  // end namespace ML

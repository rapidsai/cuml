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

#include "cuML_raft_comms_impl.hpp"

#include <raft/comms/helper.hpp>


#include <common/cumlHandle.hpp>
#include <common/raftHandle_impl.hpp>
#include <cuML_comms.hpp>

#include <cuml/common/logger.hpp>

#include <common/cudart_utils.h>

namespace ML {

namespace {

size_t getDatatypeSize(const cumlRAFTCommunicator_impl::datatype_t datatype) {
  return raft::comms::get_datatype_size((raft::comms::datatype_t) datatype);
}

ncclDataType_t getNCCLDatatype(
  const cumlRAFTCommunicator_impl::datatype_t datatype) {
    return raft::comms::get_nccl_datatype((raft::comms::datatype_t) datatype);
}

ncclRedOp_t getNCCLOp(const cumlRAFTCommunicator_impl::op_t op) {
  return raft::comms::get_nccl_op((raft::comms::op_t) op);
}
}  // namespace

/**
 * @brief Underlying comms, like NCCL and UCX, should be initialized and ready for use,
 * and maintained, outside of the cuML Comms lifecycle. This allows us to decouple the
 * ownership of the actual comms from cuml so that they can also be used outside of cuml.
 *
 * For instance, nccl-py can be used to bootstrap a ncclComm_t before it is
 * used to construct a cuml comms instance. UCX endpoints can be bootstrapped
 * in Python using ucx-py, before being used to construct a cuML comms instance.
 */
void inject_comms(cumlHandle &handle, ncclComm_t comm, ucp_worker_h ucp_worker,
                  std::shared_ptr<ucp_ep_h *> eps, int size, int rank) {
  
  auto& h_impl = cast_handle_impl(handle.getImpl());

  auto communicator = std::make_shared<MLCommon::cumlCommunicator>(
    std::unique_ptr<MLCommon::cumlCommunicator_iface>(
      new cumlRAFTCommunicator_impl(comm, ucp_worker, eps, size, rank, h_impl.getRaftHandle().get_device_allocator(), h_impl.getStream())));
  h_impl.setCommunicator(communicator);

  auto raftCommunicator = cast_comms(communicator->getImpl()).getRaftComms();

  h_impl.getRaftHandle().set_comms(raftCommunicator);
}

void inject_comms(cumlHandle &handle, ncclComm_t comm, int size, int rank) {
  
  auto& h_impl = cast_handle_impl(handle.getImpl());

  auto communicator = std::make_shared<MLCommon::cumlCommunicator>(
    std::unique_ptr<MLCommon::cumlCommunicator_iface>(
      new cumlRAFTCommunicator_impl(comm, size, rank, h_impl.getRaftHandle().get_device_allocator(), h_impl.getStream())));
  h_impl.setCommunicator(communicator);

  auto raftCommunicator = cast_comms(communicator->getImpl()).getRaftComms();

  h_impl.getRaftHandle().set_comms(raftCommunicator);
}

void inject_comms_py_coll(cumlHandle *handle, ncclComm_t comm, int size,
                          int rank) {
  inject_comms(*handle, comm, size, rank);
}

void inject_comms_py(ML::cumlHandle *handle, ncclComm_t comm, void *ucp_worker,
                     void *eps, int size, int rank) {
    #ifdef WITH_UCX
    std::shared_ptr<ucp_ep_h *> eps_sp =
        std::make_shared<ucp_ep_h *>(new ucp_ep_h[size]);

    size_t *size_t_ep_arr = (size_t *)eps;

    for (int i = 0; i < size; i++) {
        size_t ptr = size_t_ep_arr[i];
        ucp_ep_h *ucp_ep_v = (ucp_ep_h *)*eps_sp;

        if (ptr != 0) {
        ucp_ep_h eps_ptr = (ucp_ep_h)size_t_ep_arr[i];
        ucp_ep_v[i] = eps_ptr;
        } else {
        ucp_ep_v[i] = nullptr;
        }
    }

    inject_comms(*handle, comm, (ucp_worker_h)ucp_worker, eps_sp, size, rank);
    #else
    inject_comms(*handle, comm, size, rank);
    #endif
}

void ncclUniqueIdFromChar(ncclUniqueId *id, char *uniqueId, int size) {
  raft::comms::nccl_unique_id_from_char(id, uniqueId, size);
}

void get_unique_id(char *uid, int size) {
  raft::comms::get_unique_id(uid, size);
}

cumlRAFTCommunicator_impl::cumlRAFTCommunicator_impl(
  ncclComm_t comm, ucp_worker_h ucp_worker, std::shared_ptr<ucp_ep_h *> eps,
  int size, int rank, std::shared_ptr<raft::mr::device::allocator> device_allocator, cudaStream_t stream) {
    _raftComms = std::make_shared<raft::comms::comms_t>(std::unique_ptr<raft::comms::comms_iface>(new raft::comms::std_comms(comm, ucp_worker, eps, size, rank, device_allocator, stream)));
}

cumlRAFTCommunicator_impl::cumlRAFTCommunicator_impl(ncclComm_t comm, int size,
                                                   int rank, std::shared_ptr<raft::mr::device::allocator> device_allocator, cudaStream_t stream) {
    _raftComms = std::make_shared<raft::comms::comms_t>(std::unique_ptr<raft::comms::comms_iface>(new raft::comms::std_comms(comm, size, rank, device_allocator, stream)));
}

cumlRAFTCommunicator_impl::~cumlRAFTCommunicator_impl() {
}

int cumlRAFTCommunicator_impl::getSize() const { return _raftComms->get_size(); }

int cumlRAFTCommunicator_impl::getRank() const { return _raftComms->get_rank(); }

std::unique_ptr<MLCommon::cumlCommunicator_iface>
cumlRAFTCommunicator_impl::commSplit(int color, int key) const {
    _raftComms->comm_split(color, key);
}

void cumlRAFTCommunicator_impl::barrier() const {
    _raftComms->barrier();
}

void cumlRAFTCommunicator_impl::get_request_id(request_t *req) const {

}

void cumlRAFTCommunicator_impl::isend(const void *buf, int size, int dest,
                                     int tag, request_t *request) const {
    _raftComms->isend(buf, size, dest, tag, request);
}

void cumlRAFTCommunicator_impl::irecv(void *buf, int size, int source, int tag,
                                     request_t *request) const {
    _raftComms->irecv(buf, size, source, tag, request);
}

void cumlRAFTCommunicator_impl::waitall(int count,
                                       request_t array_of_requests[]) const {
    _raftComms->waitall(count, array_of_requests);
}

void cumlRAFTCommunicator_impl::allreduce(const void *sendbuff, void *recvbuff,
                                         int count, datatype_t datatype,
                                         op_t op, cudaStream_t stream) const {
    _raftComms->allreduce(sendbuff, recvbuff, count, (raft::comms::op_t) op, stream);
}

void cumlRAFTCommunicator_impl::bcast(void *buff, int count, datatype_t datatype,
                                     int root, cudaStream_t stream) const {
    _raftComms->bcast(buff, count, root, stream);
}

void cumlRAFTCommunicator_impl::reduce(const void *sendbuff, void *recvbuff,
                                      int count, datatype_t datatype, op_t op,
                                      int root, cudaStream_t stream) const {
    _raftComms->reduce(sendbuff, recvbuff, count, (raft::comms::op_t) op, root, stream);
}

void cumlRAFTCommunicator_impl::allgather(const void *sendbuff, void *recvbuff,
                                         int sendcount, datatype_t datatype,
                                         cudaStream_t stream) const {
    _raftComms->allgather(sendbuff, recvbuff, sendcount, stream);
}

void cumlRAFTCommunicator_impl::allgatherv(const void *sendbuf, void *recvbuf,
                                          const int recvcounts[],
                                          const int displs[],
                                          datatype_t datatype,
                                          cudaStream_t stream) const {
    _raftComms->allgatherv(sendbuf, recvbuf, recvcounts, displs, stream);
}

void cumlRAFTCommunicator_impl::reducescatter(const void *sendbuff,
                                             void *recvbuff, int recvcount,
                                             datatype_t datatype, op_t op,
                                             cudaStream_t stream) const {
    _raftComms->reducescatter(sendbuff, recvbuff, recvcount, (raft::comms::op_t) op, stream);
}

MLCommon::cumlCommunicator::status_t cumlRAFTCommunicator_impl::syncStream(
  cudaStream_t stream) const {
    _raftComms->sync_stream(stream);
}

std::shared_ptr<raft::comms::comms_t> cumlRAFTCommunicator_impl::getRaftComms() const {
    return _raftComms;
}

}  // end namespace ML

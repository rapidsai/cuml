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

#include "cuML_std_comms_impl.hpp"

#include <nccl.h>

#ifdef WITH_UCX
constexpr bool UCX_ENABLED = true;
#else
constexpr bool UCX_ENABLED = false;
#endif

#ifdef WITH_UCX
#include <pthread.h>
#include <ucp/api/ucp.h>
#include <ucp/api/ucp_def.h>
#include "ucp_helper.h"
#endif

#include <cstdio>
#include <exception>
#include <memory>
#include <algorithm>

#include <common/cumlHandle.hpp>
#include <cuML_comms.hpp>

#include <cuda_runtime.h>

#include <utils.h>

#define NCCL_CHECK(call)                                                       \
  do {                                                                         \
    ncclResult_t status = call;                                                \
    ASSERT(ncclSuccess == status, "ERROR: NCCL call='%s'. Reason:%s\n", #call, \
           ncclGetErrorString(status));                                        \
  } while (0)

//@todo adapt logging infrastructure for NCCL_CHECK_NO_THROW once available:
//https://github.com/rapidsai/cuml/issues/100
#define NCCL_CHECK_NO_THROW(call)                                              \
  do {                                                                         \
    ncclResult_t status = call;                                                \
    if (ncclSuccess != status) {                                               \
      std::fprintf(stderr,                                                     \
                   "ERROR: NCCL call='%s' at file=%s line=%d failed with %s ", \
                   #call, __FILE__, __LINE__, ncclGetErrorString(status));     \
    }                                                                          \
  } while (0)

namespace ML {

namespace {

size_t getDatatypeSize(const cumlStdCommunicator_impl::datatype_t datatype) {
  switch (datatype) {
    case MLCommon::cumlCommunicator::CHAR:
      return sizeof(char);
    case MLCommon::cumlCommunicator::UINT8:
      return sizeof(unsigned char);
    case MLCommon::cumlCommunicator::INT:
      return sizeof(int);
    case MLCommon::cumlCommunicator::UINT:
      return sizeof(unsigned int);
    case MLCommon::cumlCommunicator::INT64:
      return sizeof(long long int);
    case MLCommon::cumlCommunicator::UINT64:
      return sizeof(unsigned long long int);
    case MLCommon::cumlCommunicator::FLOAT:
      return sizeof(float);
    case MLCommon::cumlCommunicator::DOUBLE:
      return sizeof(double);
  }
}

ncclDataType_t getNCCLDatatype(
  const cumlStdCommunicator_impl::datatype_t datatype) {
  switch (datatype) {
    case MLCommon::cumlCommunicator::CHAR:
      return ncclChar;
    case MLCommon::cumlCommunicator::UINT8:
      return ncclUint8;
    case MLCommon::cumlCommunicator::INT:
      return ncclInt;
    case MLCommon::cumlCommunicator::UINT:
      return ncclUint32;
    case MLCommon::cumlCommunicator::INT64:
      return ncclInt64;
    case MLCommon::cumlCommunicator::UINT64:
      return ncclUint64;
    case MLCommon::cumlCommunicator::FLOAT:
      return ncclFloat;
    case MLCommon::cumlCommunicator::DOUBLE:
      return ncclDouble;
  }
}

ncclRedOp_t getNCCLOp(const cumlStdCommunicator_impl::op_t op) {
  switch (op) {
    case MLCommon::cumlCommunicator::SUM:
      return ncclSum;
    case MLCommon::cumlCommunicator::PROD:
      return ncclProd;
    case MLCommon::cumlCommunicator::MIN:
      return ncclMin;
    case MLCommon::cumlCommunicator::MAX:
      return ncclMax;
  }
}
}  // namespace

bool ucx_enabled() { return UCX_ENABLED; }

/**
 * @brief Underlying comms, like NCCL and UCX, should be initialized and ready for use,
 * and maintained, outside of the cuML Comms lifecycle. This allows us to decouple the
 * ownership of the actual comms from cuml so that they can also be used outside of cuml.
 *
 * For instance, nccl-py can be used to bootstrap a ncclComm_t before it is
 * used to construct a cuml comms instance. UCX endpoints can be bootstrapped
 * in Python using ucx-py, before being used to construct a cuML comms instance.
 */
#ifdef WITH_UCX
void inject_comms(cumlHandle &handle, ncclComm_t comm, ucp_worker_h ucp_worker,
                  std::shared_ptr<ucp_ep_h *> eps, int size, int rank) {
  auto communicator = std::make_shared<MLCommon::cumlCommunicator>(
    std::unique_ptr<MLCommon::cumlCommunicator_iface>(
      new cumlStdCommunicator_impl(comm, ucp_worker, eps, size, rank)));
  handle.getImpl().setCommunicator(communicator);
}
#endif

void inject_comms(cumlHandle &handle, ncclComm_t comm, int size, int rank) {
  auto communicator = std::make_shared<MLCommon::cumlCommunicator>(
    std::unique_ptr<MLCommon::cumlCommunicator_iface>(
      new cumlStdCommunicator_impl(comm, size, rank)));
  handle.getImpl().setCommunicator(communicator);
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
    ucp_ep_h *ucp_ep_v = *eps_sp;

    if (ptr != 0) {
      ucp_ep_h *eps_ptr = (ucp_ep_h *)size_t_ep_arr[i];
      ucp_ep_v[i] = *eps_ptr;
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
  memcpy(id->internal, uniqueId, size);
}

/**
 * @brief Returns a NCCL unique ID as a character array. PyTorch
 * uses this same approach, so that it can be more easily
 * converted to a native Python string by Cython and further
 * serialized to be sent across process & node boundaries.
 *
 * @returns the generated NCCL unique ID for establishing a
 * new clique.
 */
void get_unique_id(char *uid, int size) {
  ncclUniqueId id;
  ncclGetUniqueId(&id);

  memcpy(uid, id.internal, size);
}

#ifdef WITH_UCX
cumlStdCommunicator_impl::cumlStdCommunicator_impl(
  ncclComm_t comm, ucp_worker_h ucp_worker, std::shared_ptr<ucp_ep_h *> eps,
  int size, int rank)
  : _nccl_comm(comm),
    _ucp_worker(ucp_worker),
    _ucp_eps(eps),
    _size(size),
    _rank(rank),
    _next_request_id(0) {
  initialize();
}
#endif

cumlStdCommunicator_impl::cumlStdCommunicator_impl(ncclComm_t comm, int size,
                                                   int rank)
  : _nccl_comm(comm), _size(size), _rank(rank) {
  initialize();
}

void cumlStdCommunicator_impl::initialize() {
  cudaStreamCreate(&stream);

  CUDA_CHECK(cudaMalloc((void **)&sendbuff, sizeof(int)));
  CUDA_CHECK(cudaMalloc((void **)&recvbuff, sizeof(int)));
}

cumlStdCommunicator_impl::~cumlStdCommunicator_impl() {

  cudaStreamDestroy(stream);

  CUDA_CHECK_NO_THROW(cudaFree(sendbuff));
  CUDA_CHECK_NO_THROW(cudaFree(recvbuff));
}

int cumlStdCommunicator_impl::getSize() const { return _size; }

int cumlStdCommunicator_impl::getRank() const { return _rank; }

std::unique_ptr<MLCommon::cumlCommunicator_iface>
cumlStdCommunicator_impl::commSplit(int color, int key) const {
  // Not supported by NCCL
  ASSERT(
    false,
    "ERROR: commSplit called but not yet supported in this comms implementation.");
}

void cumlStdCommunicator_impl::barrier() const {
  CUDA_CHECK(cudaMemsetAsync(sendbuff, 0, sizeof(int), stream));
  CUDA_CHECK(cudaMemsetAsync(recvbuff, 0, sizeof(int), stream));

  allreduce(sendbuff, recvbuff, 1, MLCommon::cumlCommunicator::INT,
            MLCommon::cumlCommunicator::SUM, stream);

  cudaStreamSynchronize(stream);
}

void cumlStdCommunicator_impl::get_request_id(request_t *req) const {

#ifdef WITH_UCX

  request_t req_id;

  if (this->_free_requests.empty())
    req_id = this->_next_request_id++;
  else {
    auto it = this->_free_requests.begin();
    req_id = *it;
    this->_free_requests.erase(it);
  }
  *req = req_id;
#endif
}

void cumlStdCommunicator_impl::isend(const void *buf, int size, int dest,
                                     int tag, request_t *request) const {
#ifdef WITH_UCX
  ASSERT(_ucp_worker != nullptr,
         "ERROR: UCX comms not initialized on communicator.");

  get_request_id(request);

  ucp_ep_h ep_ptr = (*_ucp_eps)[dest];

  struct ucx_context *ucp_request = ucp_isend(ep_ptr, buf, size, tag, getRank());

  _requests_in_flight.insert(std::make_pair(*request, ucp_request));
#endif

  ASSERT(UCX_ENABLED, "cuML Comms not built with UCX support");
}



void cumlStdCommunicator_impl::irecv(void *buf, int size, int source, int tag,
                                     request_t *request) const {
#ifdef WITH_UCX
  ASSERT(_ucp_worker != nullptr,
         "ERROR: UCX comms not initialized on communicator.");

  get_request_id(request);

  ucp_ep_h ep_ptr = (*_ucp_eps)[source];

  struct ucx_context *ucp_request =
    ucp_irecv(_ucp_worker, ep_ptr, buf, size, tag, source);

  _requests_in_flight.insert(std::make_pair(*request, ucp_request));
#endif

  ASSERT(UCX_ENABLED, "cuML Comms not built with UCX support");
}

void cumlStdCommunicator_impl::waitall(int count,
                                       request_t array_of_requests[]) const {
#ifdef WITH_UCX
  ASSERT(_ucp_worker != nullptr,
         "ERROR: UCX comms not initialized on communicator.");

  std::vector<struct ucx_context *> requests;
  requests.reserve(count);

  for (int i = 0; i < count; ++i) {
    auto req_it = _requests_in_flight.find(array_of_requests[i]);
    ASSERT(_requests_in_flight.end() != req_it,
           "ERROR: waitall on invalid request: %d", array_of_requests[i]);
    requests.push_back(req_it->second);
    _free_requests.insert(req_it->first);
    _requests_in_flight.erase(req_it);
  }

  std::vector<struct ucx_context *> done_requests;
  done_requests.reserve(count);

  while(done_requests.size() < count) {

    for (struct ucx_context *req : requests) {

      ucp_worker_progress(_ucp_worker);

      auto req_it = std::find(done_requests.begin(), done_requests.end(), req);
      if(req->completed == 1 && req_it == done_requests.end())
        done_requests.push_back(req);
    }
  }

  for(struct ucx_context *req : done_requests) {
    req->completed = 0;
    if (req->needs_release) ucp_request_free(req);
  }

#endif

  ASSERT(UCX_ENABLED, "cuML Comms not built with UCX support");
}

void cumlStdCommunicator_impl::allreduce(const void *sendbuff, void *recvbuff,
                                         int count, datatype_t datatype,
                                         op_t op, cudaStream_t stream) const {
  NCCL_CHECK(ncclAllReduce(sendbuff, recvbuff, count, getNCCLDatatype(datatype),
                           getNCCLOp(op), _nccl_comm, stream));
}

void cumlStdCommunicator_impl::bcast(void *buff, int count, datatype_t datatype,
                                     int root, cudaStream_t stream) const {
  NCCL_CHECK(ncclBroadcast(buff, buff, count, getNCCLDatatype(datatype), root,
                           _nccl_comm, stream));
}

void cumlStdCommunicator_impl::reduce(const void *sendbuff, void *recvbuff,
                                      int count, datatype_t datatype, op_t op,
                                      int root, cudaStream_t stream) const {
  NCCL_CHECK(ncclReduce(sendbuff, recvbuff, count, getNCCLDatatype(datatype),
                        getNCCLOp(op), root, _nccl_comm, stream));
}

void cumlStdCommunicator_impl::allgather(const void *sendbuff, void *recvbuff,
                                         int sendcount, datatype_t datatype,
                                         cudaStream_t stream) const {
  NCCL_CHECK(ncclAllGather(sendbuff, recvbuff, sendcount,
                           getNCCLDatatype(datatype), _nccl_comm, stream));
}

void cumlStdCommunicator_impl::allgatherv(const void *sendbuf, void *recvbuf,
                                          const int recvcounts[],
                                          const int displs[],
                                          datatype_t datatype,
                                          cudaStream_t stream) const {
  //From: "An Empirical Evaluation of Allgatherv on Multi-GPU Systems" - https://arxiv.org/pdf/1812.05964.pdf
  //Listing 1 on page 4.
  for (int root = 0; root < _size; ++root)
    NCCL_CHECK(ncclBroadcast(
      sendbuf,
      static_cast<char *>(recvbuf) + displs[root] * getDatatypeSize(datatype),
      recvcounts[root], getNCCLDatatype(datatype), root, _nccl_comm, stream));
}

void cumlStdCommunicator_impl::reducescatter(const void *sendbuff,
                                             void *recvbuff, int recvcount,
                                             datatype_t datatype, op_t op,
                                             cudaStream_t stream) const {
  NCCL_CHECK(ncclReduceScatter(sendbuff, recvbuff, recvcount,
                               getNCCLDatatype(datatype), getNCCLOp(op),
                               _nccl_comm, stream));
}

}  // end namespace ML

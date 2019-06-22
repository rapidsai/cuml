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

#include "cuML_comms_nccl_impl.hpp"

#include <nccl.h>
#include <ucp/api/ucp.h>
#include <ucp/api/ucp_def.h>

#include <memory>
#include <cstdio>
#include <exception>
#include <pthread.h>

#include <cuML_comms.hpp>
#include <common/cumlHandle.hpp>

#include <utils.h>


#define NCCL_CHECK(call)                                                        \
  do {                                                                          \
    ncclResult_t status = call;                                                 \
    ASSERT(ncclSuccess == status, "ERROR: NCCL call='%s'. Reason:%s\n", #call,  \
        ncclGetErrorString(status));                                            \
  } while(0)

//@todo adapt logging infrastructure for NCCL_CHECK_NO_THROW once available:
//https://github.com/rapidsai/cuml/issues/100
#define NCCL_CHECK_NO_THROW(call)                                       \
  do {                                                                  \
    ncclResult_t status = call;                                         \
    if ( ncclSuccess != status ) {                                      \
      std::fprintf(stderr,                                              \
          "ERROR: NCCL call='%s' at file=%s line=%d failed with %s ",   \
          #call, __FILE__, __LINE__, ncclGetErrorString(status) );      \
    }                                                                   \
  } while(0)

namespace ML {

namespace {


    size_t getDatatypeSize( const cumlNCCLCommunicator_impl::datatype_t datatype )
    {
        switch ( datatype )
        {
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

    ncclDataType_t getNCCLDatatype( const cumlNCCLCommunicator_impl::datatype_t datatype )
    {
        switch ( datatype )
        {
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

    ncclRedOp_t getNCCLOp( const cumlNCCLCommunicator_impl::op_t op )
    {
        switch ( op )
        {
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
}


static const ucp_tag_t default_tag_mask = -1;

static void wait(ucp_worker_h ucp_worker, struct ucx_context *context)
{
    while (context->completed == 0) {
        ucp_worker_progress(ucp_worker);
    }
}

/**
 * Underlying comms, like NCCL and UCX, should be initialized and ready for use
 * outside of the cuML Comms lifecycle. This allows us to decouple the ownership
 * of the actual comms from cuml so that they can also be used directly, outside of
 * cuml.
 *
 * For instance, nccl-py can be used to bootstrap a ncclComm_t before it is
 * used to construct a cuml comms instance. UCX endpoints can be bootstrapped
 * in Python as well, before being used to construct a cuML comms instance.
 */
void inject_comms(cumlHandle& handle, ncclComm_t comm, ucp_worker_h ucp_worker, ucp_ep_h *eps, int size, int rank)
{
    auto communicator = std::make_shared<MLCommon::cumlCommunicator>(
         std::unique_ptr<MLCommon::cumlCommunicator_iface>( new cumlNCCLCommunicator_impl(comm, ucp_worker, eps, size, rank) ) );
    handle.getImpl().setCommunicator( communicator );
}

cumlNCCLCommunicator_impl::cumlNCCLCommunicator_impl(ncclComm_t comm, ucp_worker_h ucp_worker, ucp_ep_h *eps, int size, int rank)
    : _nccl_comm(comm), _ucp_worker(ucp_worker), _ucp_eps(eps), _size(size), _rank(rank), _next_request_id(0) {
    //initializing NCCL
//    NCCL_CHECK(ncclCommInitRank(&_nccl_comm, _size, _rank));
}

cumlNCCLCommunicator_impl::~cumlNCCLCommunicator_impl() {}

int cumlNCCLCommunicator_impl::getSize() const {
    return _size;
}

int cumlNCCLCommunicator_impl::getRank() const {
    return _rank;
}

std::unique_ptr<MLCommon::cumlCommunicator_iface> cumlNCCLCommunicator_impl::commSplit( int color, int key ) const
{
    // Not supported by NCCL
    printf("commSplit called but not supported in NCCL implementation.\n");
}

void cumlNCCLCommunicator_impl::barrier() const
{
    // @TODO:
    // 1. Have Rank 0 send out predetermined message and blocks until it gets a response from everyone.
    // 2. All other ranks block until they see the message and reply with a predetermined response.
    // 3. Upon getting responses from everyone, Rank 0 sends out a message for everyone to continue.
    printf("barrier called but not supported in NCCL implementation.\n");
}

static void send_handle(void *request, ucs_status_t status) {


    //pthread_mutex_lock(&m);
    printf("INSIDE SEND HANDLE!\n");
    struct ucx_context *context = (struct ucx_context *) request;
    context->completed = 1;

    printf("Finished in send handle\n");

    printf("[0x%x] send handler called with status %d (%s)\n",
           (unsigned int)pthread_self(), status, ucs_status_string(status));

   // pthread_mutex_unlock(&m);
}

static void recv_handle(void *request, ucs_status_t status,
                        ucp_tag_recv_info_t *info) {

    //pthread_mutex_lock(&m);

    printf("INSIDE RECEIVE HANDLE!\n");
    struct ucx_context *context = (struct ucx_context *) request;
    context->completed = 1;
    printf("Finished in receive handle\n");


    printf("[0x%x] receive handler called with status %d (%s), length %lu\n",
           (unsigned int)pthread_self(), status, ucs_status_string(status),
           info->length);

    //pthread_mutex_unlock(&m);
}

static void flush_callback(void *request, ucs_status_t status){}

static ucs_status_t flush_ep(ucp_worker_h worker, ucp_ep_h ep)
{
    void *request;

    request = ucp_ep_flush_nb(ep, 0, flush_callback);
    if (request == NULL) {
        return UCS_OK;
    } else if (UCS_PTR_IS_ERR(request)) {
        return UCS_PTR_STATUS(request);
    } else {
        ucs_status_t status;
        do {
            ucp_worker_progress(worker);
            status = ucp_request_check_status(request);
        } while (status == UCS_INPROGRESS);
        ucp_request_release(request);
        return status;
    }
}

// @TODO: Really, the isend and irecv should be tied to a datatype, as they are in MPI/UCP.
// This may not matter for p2p, though, if we are just always going to treat them as a contiguous sequence of bytes.
void cumlNCCLCommunicator_impl::isend(const void *buf, int size, int dest, int tag, request_t *request) const
{

  request_t req_id;
  if ( _free_requests.empty() )
      req_id = _next_request_id++;
  else {
      auto it = _free_requests.begin();
      req_id = *it;
      _free_requests.erase(it);
      
  }

  struct ucx_context *ucp_request = 0;
  ucp_tag_t ucp_tag = (ucp_tag_t)tag;
  ucp_ep_h ep_ptr = _ucp_eps[dest];


//  ucp_worker_print_info (_ucp_worker, stdout);
  
//  ucp_ep_print_info(ep_ptr, stdout);


  ucp_request = (struct ucx_context*)ucp_tag_send_nb(ep_ptr, buf, size,
                              ucp_dt_make_contig(1), ucp_tag, send_handle);

  if(UCS_PTR_STATUS(ucp_request) == UCS_OK)
      printf("It's null already!\n");

   if (UCS_PTR_IS_ERR(ucp_request)) {
       printf("unable to send UCX data message\n");
       ucp_ep_close_nb(ep_ptr, UCP_EP_CLOSE_MODE_FLUSH);
       return;
   } else if (UCS_PTR_STATUS(ucp_request) != UCS_OK) {
       // wait(_ucp_worker, ucp_request);
       // ucp_request->completed = 0; /* Reset request state before recycling it */
       // ucp_request_release(ucp_request);

       printf("An error occurred sending message.\n");
    } else {
        //request is complete so no need to wait on request
        ucp_request = (struct ucx_context*)malloc(sizeof(struct ucx_context));
        ucp_request->completed = 1;
        ucp_request->needs_release = false;
    }


    if(ucp_request == nullptr)
        printf("The request on rank %d was NULL!\n", getRank());

    //pthread_mutex_lock(&m);
    _requests_in_flight.insert( std::make_pair( req_id, ucp_request ) );
    *request = req_id;
    //pthread_mutex_unlock(&m);

    ucs_status_t flush_status = flush_ep(_ucp_worker, ep_ptr);
    printf("flush_ep completed with status %d (%s)\n",
            flush_status, ucs_status_string(flush_status));
}

void cumlNCCLCommunicator_impl::irecv(void *buf, int size, int source, int tag, request_t *request) const
{

  request_t req_id;
  if ( _free_requests.empty() )
      req_id = _next_request_id++;
  else {
      auto it = _free_requests.begin();
      req_id = *it;
      _free_requests.erase(it);
  }

  struct ucx_context *ucp_request = 0;
  ucp_ep_h ep_ptr = _ucp_eps[source];
  ucp_tag_t ucp_tag = (ucp_tag_t)tag;


 // std::cout << "RECV EP_PTRE: " << ep_ptr << std::endl;
//  std::cout << "UCP WORKER: " << _ucp_worker << std::endl;


  ucp_request = (struct ucx_context*)ucp_tag_recv_nb(_ucp_worker, buf, size,
                            ucp_dt_make_contig(1), ucp_tag, default_tag_mask,
                            recv_handle);



  if (UCS_PTR_IS_ERR(ucp_request)) {
      printf("unable to receive UCX data message (%d)\n");
       //       UCS_PTR_STATUS(ucp_request));
      ucp_ep_close_nb(ep_ptr, UCP_EP_CLOSE_MODE_FLUSH);
      return;
  } else {

    
    //wait(_ucp_worker, ucp_request);
    //ucp_request->completed = 0;
    //ucp_request_release(request);

    printf("Cleaned up request on %d\n", getRank());
}

  //pthread_mutex_lock(&m);
  _requests_in_flight.insert( std::make_pair( req_id, ucp_request ) );
  *request = req_id;
  //pthread_mutex_unlock(&m);
}

void cumlNCCLCommunicator_impl::waitall(int count, request_t array_of_requests[]) const
{

  printf("Inside waitall for rank: %d\n", getRank());
  std::vector<struct ucx_context*> requests;
  for ( int i = 0; i < count; ++i ) {
       auto req_it = _requests_in_flight.find( array_of_requests[i] );
       ASSERT( _requests_in_flight.end() != req_it, "ERROR: waitall on invalid request: %d", array_of_requests[i] );


       if(req_it->second == nullptr)
           printf("Encountered null request on rank %d\n", getRank());

       printf("Adding request to request on rank %d\n", getRank());
       requests.push_back( req_it->second );

       printf("Inserting request into _free_requests on rank %d\n", getRank());
       _free_requests.insert( req_it->first );
    
      printf("Erasing from requests in flight on rank %d\n", getRank());
      _requests_in_flight.erase( req_it );
  }

  int done = 0;


  printf("Checking completed on rank %d\n", getRank());



  done = 0;
  for(struct ucx_context *req : requests) {

      if(req == nullptr) {
          printf("Encountered null request on rank %d\n", getRank());
          continue;
      }

      wait(_ucp_worker, req);

      //pthread_mutex_lock(&m);
      req->completed = 0; /* Reset request state before recycling it */
      //pthread_mutex_unlock(&m);

      if(req->needs_release)
          ucp_request_release(req);
      printf("Checked off request on rank %d\n", getRank());
      

  }

  printf("Done waitall for rank: %d\n", getRank());
}

void cumlNCCLCommunicator_impl::allreduce(const void* sendbuff, void* recvbuff, int count, datatype_t datatype, op_t op, cudaStream_t stream) const
{
    NCCL_CHECK( ncclAllReduce(sendbuff, recvbuff, count, getNCCLDatatype( datatype ), getNCCLOp( op ), _nccl_comm, stream) );
}

void cumlNCCLCommunicator_impl::bcast(void* buff, int count, datatype_t datatype, int root, cudaStream_t stream) const
{
    NCCL_CHECK( ncclBroadcast(buff, buff, count, getNCCLDatatype( datatype ), root, _nccl_comm, stream) );
}

void cumlNCCLCommunicator_impl::reduce(const void* sendbuff, void* recvbuff, int count, datatype_t datatype, op_t op, int root, cudaStream_t stream) const
{
    NCCL_CHECK( ncclReduce(sendbuff, recvbuff, count, getNCCLDatatype( datatype ), getNCCLOp( op ), root, _nccl_comm, stream) );
}

void cumlNCCLCommunicator_impl::allgather(const void* sendbuff, void* recvbuff, int sendcount, datatype_t datatype, cudaStream_t stream) const
{
    NCCL_CHECK( ncclAllGather(sendbuff, recvbuff, sendcount, getNCCLDatatype( datatype ), _nccl_comm, stream) );
}

void cumlNCCLCommunicator_impl::allgatherv(const void *sendbuf, void *recvbuf, const int recvcounts[], const int displs[], datatype_t datatype, cudaStream_t stream) const
{
    //From: "An Empirical Evaluation of Allgatherv on Multi-GPU Systems" - https://arxiv.org/pdf/1812.05964.pdf
    //Listing 1 on page 4.
    for (int root = 0; root < _size; ++root)
        NCCL_CHECK( ncclBroadcast(sendbuf, static_cast<char*>(recvbuf)+displs[root]*getDatatypeSize( datatype ), recvcounts[root], getNCCLDatatype( datatype ), root, _nccl_comm, stream) );
}

void cumlNCCLCommunicator_impl::reducescatter(const void* sendbuff, void* recvbuff, int recvcount, datatype_t datatype, op_t op, cudaStream_t stream) const
{
    NCCL_CHECK( ncclReduceScatter(sendbuff, recvbuff, recvcount, getNCCLDatatype( datatype ), getNCCLOp( op ), _nccl_comm, stream) );
}

} // end namespace ML

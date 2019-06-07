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

#include <memory>
#include <cstdio>

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

void initialize_comms(cumlHandle& handle, ncclComm_t comm, int size, int rank)
{
    auto communicator = std::make_shared<MLCommon::cumlCommunicator>(
         std::unique_ptr<MLCommon::cumlCommunicator_iface>( new cumlNCCLCommunicator_impl(comm, size, rank) ) );
    handle.getImpl().setCommunicator( communicator );
}

cumlNCCLCommunicator_impl::cumlNCCLCommunicator_impl(ncclComm_t comm, int size, int rank)
    : _nccl_comm(comm), _size(size), _rank(rank) {
    //initializing NCCL
//    NCCL_CHECK(ncclCommInitRank(&_nccl_comm, _size, _rank));
}

cumlNCCLCommunicator_impl::~cumlNCCLCommunicator_impl()
{
    //finalizing NCCL
    NCCL_CHECK_NO_THROW( ncclCommDestroy(_nccl_comm) );
}

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
    // not supported by NCCL
    printf("barrier called but not supported in NCCL implementation.\n");
}

void cumlNCCLCommunicator_impl::isend(const void *buf, int size, int dest, int tag, request_t *request) const
{
    // Will investigate supporting UCX for this
    printf("isend called but not supported in NCCL implementation.\n");
}

void cumlNCCLCommunicator_impl::irecv(void *buf, int size, int source, int tag, request_t *request) const
{
    // Will investigate supporting UCX for this
    printf("irecv called but not supported in NCCL implementation.\n");
}

void cumlNCCLCommunicator_impl::waitall(int count, request_t array_of_requests[]) const
{
    // Not supported by NCCL
    printf("waitall called but not supported in NCCL implementation.\n");
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

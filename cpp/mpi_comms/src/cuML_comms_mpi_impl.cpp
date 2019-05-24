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

#include "cuML_comms_mpi_impl.hpp"

#include <memory>
#include <cstdio>

#include <cuML_comms.hpp>
#include <common/cumlHandle.hpp>

#include <utils.h>

#define MPI_CHECK(call)                                                             \
  do {                                                                              \
    int status = call;                                                              \
    if ( MPI_SUCCESS != status ) {                                                  \
        int mpi_error_string_lenght = 0;                                            \
        char mpi_error_string[MPI_MAX_ERROR_STRING];                                \
        MPI_Error_string(status, mpi_error_string, &mpi_error_string_lenght);       \
        ASSERT(MPI_SUCCESS == status, "ERROR: MPI call='%s'. Reason:%s\n", #call,   \
            mpi_error_string);                                                      \
    }                                                                               \
  } while (0)

//@todo adapt logging infrastructure for NCCL_CHECK_NO_THROW once available:
//https://github.com/rapidsai/cuml/issues/100
#define MPI_CHECK_NO_THROW(call)                                            \
  do {                                                                      \
    int status = call;                                                      \
    if ( MPI_SUCCESS != status ) {                                          \
      int mpi_error_string_lenght = 0;                                      \
      char mpi_error_string[MPI_MAX_ERROR_STRING];                          \
      MPI_Error_string(status, mpi_error_string, &mpi_error_string_lenght); \
      std::fprintf(stderr,                                                  \
          "ERROR: MPI call='%s' at file=%s line=%d failed with %s ",        \
          #call, __FILE__, __LINE__, mpi_error_string );                    \
    }                                                                       \
  } while(0)

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
    size_t getDatatypeSize( const cumlMPICommunicator_impl::datatype_t datatype )
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

    MPI_Datatype getMPIDatatype( const cumlMPICommunicator_impl::datatype_t datatype )
    {
        switch ( datatype )
        {
            case MLCommon::cumlCommunicator::CHAR:
                return MPI_CHAR;
            case MLCommon::cumlCommunicator::UINT8:
                return MPI_UNSIGNED_CHAR;
            case MLCommon::cumlCommunicator::INT:
                return MPI_INT;
            case MLCommon::cumlCommunicator::UINT:
                return MPI_UNSIGNED;
            case MLCommon::cumlCommunicator::INT64:
                return MPI_LONG_LONG;
            case MLCommon::cumlCommunicator::UINT64:
                return MPI_UNSIGNED_LONG_LONG;
            case MLCommon::cumlCommunicator::FLOAT:
                return MPI_FLOAT;
            case MLCommon::cumlCommunicator::DOUBLE:
                return MPI_DOUBLE;
        }
    }

    MPI_Op getMPIOp( const cumlMPICommunicator_impl::op_t op )
    {
        switch ( op )
        {
            case MLCommon::cumlCommunicator::SUM:
                return MPI_SUM;
            case MLCommon::cumlCommunicator::PROD:
                return MPI_PROD;
            case MLCommon::cumlCommunicator::MIN:
                return MPI_MIN;
            case MLCommon::cumlCommunicator::MAX:
                return MPI_MAX;
        }
    }

#ifdef HAVE_NCCL
    ncclDataType_t getNCCLDatatype( const cumlMPICommunicator_impl::datatype_t datatype )
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

    ncclRedOp_t getNCCLOp( const cumlMPICommunicator_impl::op_t op )
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
#endif
}

void initialize_comms(cumlHandle& handle, MPI_Comm comm)
{
    auto communicator = std::make_shared<MLCommon::cumlCommunicator>( 
         std::unique_ptr<MLCommon::cumlCommunicator_iface>( new cumlMPICommunicator_impl(comm) ) );
    handle.getImpl().setCommunicator( communicator );
}

cumlMPICommunicator_impl::cumlMPICommunicator_impl(MPI_Comm comm, const bool owns_mpi_comm)
    : _owns_mpi_comm(owns_mpi_comm), _mpi_comm(comm), _size(0), _rank(1), _next_request_id(0)
{
    int mpi_is_initialized = 0;
    MPI_CHECK( MPI_Initialized(&mpi_is_initialized) );
    ASSERT( mpi_is_initialized, "ERROR: MPI is not initialized!" );
    MPI_CHECK( MPI_Comm_size( _mpi_comm, &_size ) );
    MPI_CHECK( MPI_Comm_rank( _mpi_comm, &_rank ) );
#ifdef HAVE_NCCL
    //get NCCL unique ID at rank 0 and broadcast it to all others
    ncclUniqueId id;
    if ( 0 == _rank ) NCCL_CHECK(ncclGetUniqueId(&id));
    MPI_CHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, _mpi_comm));

    //initializing NCCL
    NCCL_CHECK(ncclCommInitRank(&_nccl_comm, _size, id, _rank));
#endif
}

cumlMPICommunicator_impl::~cumlMPICommunicator_impl()
{
#ifdef HAVE_NCCL
    //finalizing NCCL
    NCCL_CHECK_NO_THROW( ncclCommDestroy(_nccl_comm) );
#endif
    if (_owns_mpi_comm)
    {
        MPI_CHECK_NO_THROW(MPI_Comm_free(&_mpi_comm));
    }
}

int cumlMPICommunicator_impl::getSize() const
{
    return _size;
}

int cumlMPICommunicator_impl::getRank() const
{
    return _rank;
}

std::unique_ptr<MLCommon::cumlCommunicator_iface> cumlMPICommunicator_impl::commSplit( int color, int key ) const
{
    MPI_Comm new_comm;
    MPI_CHECK( MPI_Comm_split(_mpi_comm, color, key, &new_comm) );
    return std::unique_ptr<MLCommon::cumlCommunicator_iface>(new cumlMPICommunicator_impl(new_comm,true));
}

void cumlMPICommunicator_impl::barrier() const
{
    MPI_CHECK( MPI_Barrier( _mpi_comm ) );
}

void cumlMPICommunicator_impl::isend(const void *buf, int size, int dest, int tag, request_t *request) const
{
    MPI_Request mpi_req;
    request_t req_id;
    if ( _free_requests.empty() )
    {
        req_id = _next_request_id++;
    }
    else
    {
        auto it = _free_requests.begin();
        req_id = *it;
        _free_requests.erase(it);
    }
    MPI_CHECK( MPI_Isend(buf, size, MPI_BYTE, dest, tag, _mpi_comm, &mpi_req) );
    _requests_in_flight.insert( std::make_pair( req_id, mpi_req ) );
    *request = req_id;
}

void cumlMPICommunicator_impl::irecv(void *buf, int size, int source, int tag, request_t *request) const
{
    MPI_Request mpi_req;
    request_t req_id;
    if ( _free_requests.empty() )
    {
        req_id = _next_request_id++;
    }
    else
    {
        auto it = _free_requests.begin();
        req_id = *it;
        _free_requests.erase(it);
    }
    MPI_CHECK( MPI_Irecv(buf, size, MPI_BYTE, source, tag, _mpi_comm, &mpi_req) );
    _requests_in_flight.insert( std::make_pair( req_id, mpi_req ) );
    *request = req_id;
}

void cumlMPICommunicator_impl::waitall(int count, request_t array_of_requests[]) const
{
    std::vector<MPI_Request> requests;
    requests.reserve(count);
    for ( int i = 0; i < count; ++i )
    {
         auto req_it = _requests_in_flight.find( array_of_requests[i] );
         ASSERT( _requests_in_flight.end() != req_it, "ERROR: waitall on invalid request: %d", array_of_requests[i] );
         requests.push_back( req_it->second );
         _free_requests.insert( req_it->first );
        _requests_in_flight.erase( req_it );
    }
    MPI_CHECK( MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE) );
}

void cumlMPICommunicator_impl::allreduce(const void* sendbuff, void* recvbuff, int count, datatype_t datatype, op_t op, cudaStream_t stream) const
{
#ifdef HAVE_NCCL
    NCCL_CHECK( ncclAllReduce(sendbuff, recvbuff, count, getNCCLDatatype( datatype ), getNCCLOp( op ), _nccl_comm, stream) );
#else
    CUDA_CHECK( cudaStreamSynchronize( stream ) );
    MPI_CHECK( MPI_Allreduce(sendbuff, recvbuff, count, getMPIDatatype( datatype ), getMPIOp( op ), _mpi_comm) );
#endif
}

void cumlMPICommunicator_impl::bcast(void* buff, int count, datatype_t datatype, int root, cudaStream_t stream) const
{
#ifdef HAVE_NCCL
    NCCL_CHECK( ncclBroadcast(buff, buff, count, getNCCLDatatype( datatype ), root, _nccl_comm, stream) );
#else
    CUDA_CHECK( cudaStreamSynchronize( stream ) );
    MPI_CHECK( MPI_Bcast(buff, count, getMPIDatatype( datatype ), root, _mpi_comm) );
#endif
}

void cumlMPICommunicator_impl::reduce(const void* sendbuff, void* recvbuff, int count, datatype_t datatype, op_t op, int root, cudaStream_t stream) const
{
#ifdef HAVE_NCCL
    NCCL_CHECK( ncclReduce(sendbuff, recvbuff, count, getNCCLDatatype( datatype ), getNCCLOp( op ), root, _nccl_comm, stream) );
#else
    CUDA_CHECK( cudaStreamSynchronize( stream ) );
    MPI_CHECK( MPI_Reduce(sendbuff, recvbuff, count, getMPIDatatype( datatype ), getMPIOp( op ), root, _mpi_comm) );
#endif
}

void cumlMPICommunicator_impl::allgather(const void* sendbuff, void* recvbuff, int sendcount, datatype_t datatype, cudaStream_t stream) const
{
#ifdef HAVE_NCCL
    NCCL_CHECK( ncclAllGather(sendbuff, recvbuff, sendcount, getNCCLDatatype( datatype ), _nccl_comm, stream) );
#else
    CUDA_CHECK( cudaStreamSynchronize( stream ) );
    MPI_CHECK( MPI_Allgather(sendbuff, sendcount, getMPIDatatype( datatype ), recvbuff, sendcount, getMPIDatatype( datatype ), _mpi_comm) );
#endif
}

void cumlMPICommunicator_impl::allgatherv(const void *sendbuf, void *recvbuf, const int recvcounts[], const int displs[], datatype_t datatype, cudaStream_t stream) const
{
#ifdef HAVE_NCCL
    //From: "An Empirical Evaluation of Allgatherv on Multi-GPU Systems" - https://arxiv.org/pdf/1812.05964.pdf
    //Listing 1 on page 4.
    for (int root = 0; root < _size; ++root) {
        NCCL_CHECK( ncclBroadcast(sendbuf, static_cast<char*>(recvbuf)+displs[root]*getDatatypeSize( datatype ), recvcounts[root], getNCCLDatatype( datatype ), root, _nccl_comm, stream) );
    }
#else
    CUDA_CHECK( cudaStreamSynchronize( stream ) );
    MPI_CHECK( MPI_Allgatherv(sendbuf, recvcounts[_rank], getMPIDatatype( datatype ), recvbuf, recvcounts, displs, getMPIDatatype( datatype ), _mpi_comm) );
#endif
}

void cumlMPICommunicator_impl::reducescatter(const void* sendbuff, void* recvbuff, int recvcount, datatype_t datatype, op_t op, cudaStream_t stream) const
{
#ifdef HAVE_NCCL
    NCCL_CHECK( ncclReduceScatter(sendbuff, recvbuff, recvcount, getNCCLDatatype( datatype ), getNCCLOp( op ), _nccl_comm, stream) );
#else
    CUDA_CHECK( cudaStreamSynchronize( stream ) );
    std::vector<int> recvcounts(_size,recvcount);
    MPI_CHECK( MPI_Reduce_scatter(sendbuff, recvbuff, recvcounts.data(), getMPIDatatype( datatype ), getMPIOp( op ), _mpi_comm) );
#endif
}

} // end namespace ML

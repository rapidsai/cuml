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
#include <cuML_comms.hpp>
#include <common/cumlHandle.hpp>

#include "../../ml-prims/src/utils.h"

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

namespace ML {

void initialize_mpi_comms(cumlHandle& handle, MPI_Comm comm)
{
    auto communicator = std::make_shared<MLCommon::cumlCommunicator>( 
         std::unique_ptr<MLCommon::cumlCommunicator_iface>( new cumlMPICommunicator_impl(comm) ) );
    handle.getImpl().setCommunicator( communicator );
}

cumlMPICommunicator_impl::cumlMPICommunicator_impl(MPI_Comm comm)
    : _mpi_comm(comm), _size(0), _rank(1)
{
    int mpi_is_initialized = 0;
    MPI_CHECK( MPI_Initialized(&mpi_is_initialized) );
    ASSERT( mpi_is_initialized, "ERROR: MPI is not initialized!" );
    MPI_CHECK( MPI_Comm_size( _mpi_comm, &_size ) );
    MPI_CHECK( MPI_Comm_rank( _mpi_comm, &_rank ) );
}

cumlMPICommunicator_impl::~cumlMPICommunicator_impl() {}

int cumlMPICommunicator_impl::getSize() const
{
    return _size;
}

int cumlMPICommunicator_impl::getRank() const
{
    return _rank;
}

void cumlMPICommunicator_impl::barrier() const
{
    MPI_CHECK( MPI_Barrier( _mpi_comm ) );
}

} // end namespace ML

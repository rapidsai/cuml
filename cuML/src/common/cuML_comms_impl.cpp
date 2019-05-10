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

#include <memory>

#include "../../../ml-prims/src/common/cuml_comms_int.hpp"
#include "../../../ml-prims/src/common/cuml_comms_iface.hpp"

#include "../../ml-prims/src/utils.h"

namespace MLCommon {

cumlCommunicator::cumlCommunicator(std::unique_ptr<cumlCommunicator_iface> impl)
    : _impl( impl.release() )
{
    ASSERT( nullptr != _impl.get(), "ERROR: Invalid cumlCommunicator_iface used!" );
}

int cumlCommunicator::getSize() const
{
    return _impl->getSize();
}

int cumlCommunicator::getRank() const
{
    return _impl->getRank();
}

void cumlCommunicator::barrier() const
{
    _impl->barrier();
}

void cumlCommunicator::isend(const void *buf, std::size_t size, int dest, int tag, request_t *request) const
{
    _impl->isend(buf, size, dest, tag, request);
}

void cumlCommunicator::irecv(void *buf, std::size_t size, int source, int tag, request_t *request) const
{
    _impl->irecv(buf, size, source, tag, request);
}

void cumlCommunicator::waitall(int count, request_t array_of_requests[]) const
{
    _impl->waitall(count, array_of_requests);
}

void cumlCommunicator::allreduce(const void* sendbuff, void* recvbuff, size_t count, datatype_t datatype, op_t op, cudaStream_t stream) const
{
    _impl->allreduce(sendbuff, recvbuff, count, datatype, op, stream);
}

template<>
cumlCommunicator::datatype_t cumlCommunicator::getDataType<char>() const
{
    return cumlCommunicator::CHAR;
}

template<>
cumlCommunicator::datatype_t cumlCommunicator::getDataType<unsigned char>() const
{
    return cumlCommunicator::UINT8;
}

template<>
cumlCommunicator::datatype_t cumlCommunicator::getDataType<int>() const
{
    return cumlCommunicator::INT;
}

template<>
cumlCommunicator::datatype_t cumlCommunicator::getDataType<unsigned int>() const
{
    return cumlCommunicator::UINT;
}

template<>
cumlCommunicator::datatype_t cumlCommunicator::getDataType<long long int>() const
{
    return cumlCommunicator::INT64;
}

template<>
cumlCommunicator::datatype_t cumlCommunicator::getDataType<unsigned long long int>() const
{
    return cumlCommunicator::UINT64;
}

template<>
cumlCommunicator::datatype_t cumlCommunicator::getDataType<float>() const
{
    return cumlCommunicator::FLOAT;
}

template<>
cumlCommunicator::datatype_t cumlCommunicator::getDataType<double>() const
{
    return cumlCommunicator::DOUBLE;
}

cumlCommunicator_iface::~cumlCommunicator_iface() {}

} // end namespace ML

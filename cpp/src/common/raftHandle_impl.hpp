#pragma once

#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cublas_v2.h>
#include <cusolverDn.h>
#include <cusolverSp.h>
#include <cusparse.h>

#include <common/cuml_comms_int.hpp>

#include <cuml/cuml.hpp>

#include <cuml/common/cuml_allocator.hpp>

#include <raft/handle.hpp>
#include "cumlHandle.hpp"

namespace ML {

using MLCommon::deviceAllocator;
using MLCommon::hostAllocator;

/**
 * @todo: Add doxygen documentation
 */
class raftHandle_impl : public cumlHandle_impl{
 public:
  raftHandle_impl(int n_streams = cumlHandle::getDefaultNumInternalStreams());
  ~raftHandle_impl();
  int getDevice() const;
  void setStream(cudaStream_t stream);
  cudaStream_t getStream() const;
  void setDeviceAllocator(std::shared_ptr<deviceAllocator> allocator);
  std::shared_ptr<deviceAllocator> getDeviceAllocator() const;
  void setHostAllocator(std::shared_ptr<hostAllocator> allocator);
  std::shared_ptr<hostAllocator> getHostAllocator() const;

  cublasHandle_t getCublasHandle() const;
  cusolverDnHandle_t getcusolverDnHandle() const;
  cusolverSpHandle_t getcusolverSpHandle() const;
  cusparseHandle_t getcusparseHandle() const;

  cudaStream_t getInternalStream(int sid) const;
  int getNumInternalStreams() const;

  std::vector<cudaStream_t> getInternalStreams() const;

  void waitOnUserStream() const;
  void waitOnInternalStreams() const;

  void setCommunicator(
    std::shared_ptr<MLCommon::cumlCommunicator> communicator);
  const MLCommon::cumlCommunicator& getCommunicator() const;
  bool commsInitialized() const;

  const cudaDeviceProp& getDeviceProperties() const;

 private:
  mutable cublasHandle_t _cublas_handle;
  mutable cusolverDnHandle_t _cusolverDn_handle;
  mutable cusolverSpHandle_t _cusolverSp_handle;
  mutable cusparseHandle_t _cusparse_handle;
  cudaStream_t _userStream;
  cudaEvent_t _event;
  std::shared_ptr<deviceAllocator> _deviceAllocator;
  std::shared_ptr<hostAllocator> _hostAllocator;
  std::shared_ptr<MLCommon::cumlCommunicator> _communicator;
  std::vector<cudaStream_t> _streams;
  mutable cudaDeviceProp _prop;
  const int _dev_id;
  const int _num_streams;
  mutable bool _cublasInitialized;
  mutable bool _cusolverDnInitialized;
  mutable bool _cusolverSpInitialized;
  mutable bool _cusparseInitialized;
  mutable bool _devicePropInitialized;

  void createResources();
  void destroyResources();
};

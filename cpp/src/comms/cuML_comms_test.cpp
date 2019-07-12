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

#include "cuML_comms_test.hpp"

#include <common/cumlHandle.hpp>
#include <common/cuml_comms_int.hpp>
#include <common/device_buffer.hpp>
#include <iostream>

namespace ML {
namespace Comms {

bool test_collective_allreduce(const ML::cumlHandle& h) {
  const cumlHandle_impl& handle = h.getImpl();
  ML::detail::streamSyncer _(handle);
  const MLCommon::cumlCommunicator& communicator = handle.getCommunicator();

  const int send = 1;

  cudaStream_t stream = handle.getStream();

  MLCommon::device_buffer<int> temp_d(handle.getDeviceAllocator(), stream);
  temp_d.resize(1, stream);
  CUDA_CHECK(cudaMemcpyAsync(temp_d.data(), &send, sizeof(int),
                             cudaMemcpyHostToDevice, stream));
  communicator.allreduce(temp_d.data(), temp_d.data(), 1,
                         MLCommon::cumlCommunicator::SUM, stream);
  int temp_h = 0;
  CUDA_CHECK(cudaMemcpyAsync(&temp_h, temp_d.data(), sizeof(int),
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  communicator.barrier();

  std::cout << "Clique size: " << communicator.getSize() << std::endl;
  std::cout << "final_size: " << temp_h << std::endl;

  return temp_h == communicator.getSize();
}

bool test_pointToPoint_simple_send_recv(const ML::cumlHandle& h,
                                        int numTrials) {
  const cumlHandle_impl& handle = h.getImpl();
  const MLCommon::cumlCommunicator& communicator = handle.getCommunicator();
  const int rank = communicator.getRank();

  for (int i = 0; i < numTrials; i++) {
    std::cout << "#############################" << std::endl;
    std::cout << "Trial " << i << " -> rank " << rank << std::endl;
    std::cout << "############################" << std::endl;

    std::vector<int> received_data((communicator.getSize() - 1), -1);

    std::vector<MLCommon::cumlCommunicator::request_t> requests;
    requests.resize(2 * (communicator.getSize() - 1));
    int request_idx = 0;
    //post receives
    for (int r = 0; r < communicator.getSize(); ++r) {
      if (r != rank) {
        communicator.irecv(received_data.data() + request_idx, 1, r, 0,
                           requests.data() + request_idx);
        ++request_idx;
      }
    }

    for (int r = 0; r < communicator.getSize(); ++r) {
      if (r != rank) {
        communicator.isend(&rank, 1, r, 0, requests.data() + request_idx);
        ++request_idx;
      }
    }

    communicator.waitall(requests.size(), requests.data());

    if (0 == rank) {
      std::cout << "There are " << communicator.getSize()
                << " ranks:" << std::endl;
    }
    communicator.barrier();

    bool ret = true;
    for (int rec : received_data) {
      std::cout << "Rank received: " << rec << std::endl;
      if (rec == -1) return false;
    }

    std::cout << "############################" << std::endl;
  }

  return true;
}

bool test_pointToPoint_simple_send_recv_cuda(const ML::cumlHandle& h,
                                        int numTrials) {
  const cumlHandle_impl& handle = h.getImpl();
  const MLCommon::cumlCommunicator& communicator = handle.getCommunicator();
  const int rank = communicator.getRank();

  for (int i = 0; i < numTrials; i++) {
    std::cout << "#############################" << std::endl;
    std::cout << "Trial " << i << " -> rank " << rank << std::endl;
    std::cout << "############################" << std::endl;

    std::vector<int> received_data((communicator.getSize() - 1), -1);

    std::vector<MLCommon::cumlCommunicator::request_t> requests;
    requests.resize(2 * (communicator.getSize() - 1));
    int request_idx = 0;
    //post receives
    for (int r = 0; r < communicator.getSize(); ++r) {
      if (r != rank) {
        communicator.irecv(received_data.data() + request_idx, 1, r, 0,
                           requests.data() + request_idx);
        ++request_idx;
      }
    }

    for (int r = 0; r < communicator.getSize(); ++r) {
      if (r != rank) {
        communicator.isend(&rank, 1, r, 0, requests.data() + request_idx);
        ++request_idx;
      }
    }

    communicator.waitall(requests.size(), requests.data());

    if (0 == rank) {
      std::cout << "There are " << communicator.getSize()
                << " ranks:" << std::endl;
    }
    communicator.barrier();

    bool ret = true;
    for (int rec : received_data) {
      std::cout << "Rank received: " << rec << std::endl;
      if (rec == -1) return false;
    }

    std::cout << "############################" << std::endl;
  }

  return true;
}


};  // namespace Comms
};  // end namespace ML

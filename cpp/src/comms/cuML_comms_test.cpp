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

#include <common/Timer.h>
#include <common/cumlHandle.hpp>
#include <common/cuml_comms_int.hpp>
#include <common/device_buffer.hpp>
#include <common/host_buffer.hpp>

#include <algorithm>
#include <cstring>
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

  bool ret = true;
  for (int i = 0; i < numTrials; i++) {
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
    communicator.barrier();

    if (communicator.getRank() == 0) {
      std::cout << "=========================" << std::endl;
      std::cout << "Trial " << i << std::endl;
    }

    for (int printrank = 0; printrank < communicator.getSize(); ++printrank) {
      if (communicator.getRank() == printrank) {
        std::cout << "Rank " << communicator.getRank() << " received: [";
        for (int i = 0; i < received_data.size(); i++) {
          auto rec = received_data[i];
          std::cout << rec;
          if (rec == -1) ret = false;
          communicator.barrier();
          if (i < received_data.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
      }

      communicator.barrier();
    }

    if (communicator.getRank() == 0)
      std::cout << "=========================" << std::endl;
  }

  return ret;
}

bool test_p2p_send_recv(const ML::cumlHandle& h, bool checkRxData,
                        bool srcIsDevice, bool dstIsDevice, int numP2pPeers,
                        int* p2pDstRankOffsets, int msg_size, int num_trials) {
  const auto& handle = h.getImpl();
  ASSERT(handle.commsInitialized() == true, "communicator uninitialized.");
  const auto& communicator = handle.getCommunicator();
  const auto rank = communicator.getRank();
  const auto comm_size = communicator.getSize();
  const auto tag = int{0};

  // check input arguments

  ASSERT((numP2pPeers > 0) && (numP2pPeers < comm_size),
         "invalid input argument: numP2pPeers=%d.", numP2pPeers);
  for (auto i = int{0}; i < numP2pPeers; i++) {
    auto offset = p2pDstRankOffsets[i];
    ASSERT((offset > 0) && (offset < comm_size),
           "invalid input argument: p2pDstRankOffsets[%d]=%d.", i, offset);
  }
  ASSERT(msg_size > 0, "invalid input argument: msg_size=%d.", msg_size);
  ASSERT(num_trials > 0, "invalid input argument: num_trials=%d.", num_trials);

  // update source & destination ranks

  std::vector<int> v_dst_rank(numP2pPeers, -1);
  std::vector<int> v_src_rank(numP2pPeers, -1);
  for (auto i = int{0}; i < numP2pPeers; i++) {
    auto offset = p2pDstRankOffsets[i];
    v_dst_rank[i] = (rank + offset) % comm_size;
    v_src_rank[i] = (rank + comm_size - offset) % comm_size;
  }

  // allocate tx & rx buffers (and tmp buffer if both checkRxData and
  // dstIsDevice are true)

  auto stream = h.getStream();

  auto&& tx_buf_d = MLCommon::device_buffer<unsigned char>(
    handle.getDeviceAllocator(), stream, 0);
  auto&& rx_buf_d = MLCommon::device_buffer<unsigned char>(
    handle.getDeviceAllocator(), stream, 0);

  auto&& tx_buf_h =
    MLCommon::host_buffer<unsigned char>(handle.getHostAllocator(), stream, 0);
  auto&& rx_buf_h =
    MLCommon::host_buffer<unsigned char>(handle.getHostAllocator(), stream, 0);

  auto&& tmp_buf_h =
    MLCommon::host_buffer<unsigned char>(handle.getHostAllocator(), stream, 0);

  if (srcIsDevice) {
    tx_buf_d.resize(msg_size * numP2pPeers, stream);
  } else {
    tx_buf_h.resize(msg_size * numP2pPeers, stream);
  }

  if (dstIsDevice) {
    rx_buf_d.resize(msg_size * numP2pPeers, stream);
    if (checkRxData) {
      tmp_buf_h.resize(msg_size, stream);
    }
  } else {
    rx_buf_h.resize(msg_size * numP2pPeers, stream);
  }

  CUDA_CHECK(cudaStreamSynchronize(stream));

  auto vp_tx_buf = std::vector<unsigned char*>(numP2pPeers, nullptr);
  if (srcIsDevice) {
    CUDA_CHECK(cudaMemset(tx_buf_d.data(), 0x00, tx_buf_d.size()));
    for (auto i = 0; i < numP2pPeers; i++) {
      vp_tx_buf[i] = tx_buf_d.data() + msg_size * i;
    }
  } else {
    memset(tx_buf_h.data(), 0x00, tx_buf_h.size());
    for (auto i = 0; i < numP2pPeers; i++) {
      vp_tx_buf[i] = tx_buf_h.data() + msg_size * i;
    }
  }

  auto vp_rx_buf = std::vector<unsigned char*>(numP2pPeers, nullptr);
  if (dstIsDevice) {
    CUDA_CHECK(cudaMemset(rx_buf_d.data(), 0x00, rx_buf_d.size()));
    for (auto i = 0; i < numP2pPeers; i++) {
      vp_rx_buf[i] = rx_buf_d.data() + msg_size * i;
    }
    if (checkRxData) {
      CUDA_CHECK(cudaMemset(tmp_buf_h.data(), 0x00, tmp_buf_h.size()));
    }
  } else {
    memset(rx_buf_h.data(), 0x00, rx_buf_h.size());
    for (auto i = 0; i < numP2pPeers; i++) {
      vp_rx_buf[i] = rx_buf_h.data() + msg_size * i;
    }
  }

  communicator.barrier();

  auto timer = MLCommon::TimerCPU{};

  // start p2p send/recieve

  for (auto trial = int{0}; trial < num_trials; trial++) {
    if (checkRxData) {
      // update tx data

      if (srcIsDevice) {
        CUDA_CHECK(cudaMemset(tx_buf_d.data(), (rank + num_trials) & 0xff,
                              tx_buf_d.size()));
      } else {
        std::memset(tx_buf_h.data(), (rank + num_trials) & 0xff,
                    tx_buf_h.size());
      }
    }

    // post rx & tx

    auto v_request = std::vector<MLCommon::cumlCommunicator::request_t>(
      2 * numP2pPeers);  // first half: tx, second half: rx

    for (auto i = int{0}; i < numP2pPeers; i++) {
      communicator.irecv(vp_rx_buf[i], msg_size, v_src_rank[i], tag,
                         &(v_request[numP2pPeers + i]));
    }

    for (auto i = int{0}; i < numP2pPeers; i++) {
      communicator.isend(vp_tx_buf[i], msg_size, v_dst_rank[i], tag,
                         &(v_request[i]));
    }

    // wait

    communicator.waitall(v_request.size(), v_request.data());

    // check rx data

    if (checkRxData) {
      auto num_err_workers = int{0};
      for (auto i = int{0}; i < numP2pPeers; i++) {
        auto p_data = static_cast<unsigned char*>(nullptr);
        if (dstIsDevice) {
          p_data = tmp_buf_h.data();
          CUDA_CHECK(cudaMemcpyAsync(p_data, vp_rx_buf[i], msg_size,
                                     cudaMemcpyDeviceToHost, stream));
          CUDA_CHECK(cudaStreamSynchronize(stream));
        } else {
          p_data = vp_rx_buf[i];
        }
        auto src_rank = v_src_rank[i];
        auto num_mismatches = std::count_if(
          p_data, p_data + msg_size, [src_rank, num_trials](unsigned char i) {
            return (i != (src_rank + num_trials) & 0xff);
          });
        if (num_mismatches > 0) {
          num_err_workers++;
        }
      }

      // this is not good. We need host memory version of allreduce as well.

      auto&& tmp_d =
        MLCommon::device_buffer<int>(handle.getDeviceAllocator(), stream, 1);
      CUDA_CHECK(cudaMemcpyAsync(tmp_d.data(), &num_err_workers, sizeof(int),
                                 cudaMemcpyHostToDevice, stream));
      communicator.allreduce(tmp_d.data(), tmp_d.data(), 1,
                             MLCommon::cumlCommunicator::SUM, stream);
      CUDA_CHECK(cudaMemcpyAsync(&num_err_workers, tmp_d.data(), sizeof(int),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaStreamSynchronize(stream));
      if (num_err_workers > 0) {
        return false;
      }
    }
  }

  communicator.barrier();

  auto elapsed_time = timer.getElapsedSeconds();

  std::cout << "Elapsed time: " << elapsed_time << " seconds." << std::endl;

  return true;
}

};  // namespace Comms
};  // end namespace ML

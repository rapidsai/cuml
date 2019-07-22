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

#pragma once

#include <cuML.hpp>

namespace ML {
namespace Comms {

/**
 * @brief Simple allreduce test for single integer value of 1. Each rank
 * evaluates whether their allreduced value equals the size of the clique.
 * @param[in] h cumlHandle instance with initialized cumlCommunicator
 */
bool test_collective_allreduce(const ML::cumlHandle& handle);

/**
 * @brief Simple point-to-point test. Each rank passes its rank to all other
 * ranks and verifies that it received messages from all other ranks.
 * @param[in] h cumlHandle instance with initialized cumlCommunicator
 * @param[in] numTrials number of iterations to pass messages
 */
bool test_pointToPoint_simple_send_recv(const ML::cumlHandle& handle,
                                        int n_trials);

/**
 * @brief Point-to-point tag matching send/receive test. Each rank send data to
 * numP2pPeers and recieve data from numP2pPeers.
 * @param[in] handle cumlHandle instance with initialized cumlCommunicator.
 * @param[in] checkRxData if set to true, perform correctness test for received
 * data. Set to false to measure send/receive performance.
 * @param[in] srcIsDevice if true source (send) data are placed in device
 * memory. If false, source data are placed in host memory.
 * @param[in] dstIsDevice if true, destination (receive) data are placed in
 * device memory. If false, destination data are placed in host memory.
 * @param[in] numP2pPeers number of peer ranks to send/receive data.
 * @param[in] p2pDstRankOffsets pointer to the array storing destination
 * rank offsets (array size: numP2pPeers).
 * @param[in] msgSize message size.
 * @param[in] numTrials number of iterations to pass messages.
 */
bool test_p2p_send_recv(const ML::cumlHandle& handle, bool checkRxData,
                        bool srcIsDevice, bool dstIsDevice, int numP2pPeers,
                        int* p2pDstRankOffsets, int msgSize, int numTrials);

};  // namespace Comms
};  // end namespace ML

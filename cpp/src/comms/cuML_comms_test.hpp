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

};  // namespace Comms
};  // end namespace ML

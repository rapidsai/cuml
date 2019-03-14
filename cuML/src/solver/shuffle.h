/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <random>

namespace ML {
namespace Solver {

using namespace MLCommon;

void initShuffle(std::vector<int> &rand_indices, std::mt19937 &g, int random_state = 0) {

	g.seed((int) random_state);
	for (int i = 0; i < rand_indices.size(); ++i)
		rand_indices[i] = i;

}

void shuffle(std::vector<int> &rand_indices, std::mt19937 &g) {
	std::shuffle(rand_indices.begin(), rand_indices.end(), g);


}

/** @} */
}
;
}
;
// end namespace ML

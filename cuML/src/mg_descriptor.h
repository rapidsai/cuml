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

#include <cstdio>
#include <stdexcept>
#include <string>

namespace ML {

class MGDescriptorFloat {


	//cudaStream_t stream;

public:

	MGDescriptorFloat(float *data, int n_rows, int n_cols):data(data), n_rows(n_rows), n_cols(n_cols) {}
	~MGDescriptorFloat() {}
	float *data;

	int n_rows;
	int n_cols;
};

class MGDescriptorDouble {

	//cudaStream_t stream;

public:

	MGDescriptorDouble(double *data, int n_rows, int n_cols):data(data), n_rows(n_rows), n_cols(n_cols) {}
	~MGDescriptorDouble() {}

	double *data;

	int n_rows;
	int n_cols;

};

} // namespace ML

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


#include "mean_c.h"
#include <stats/mean.h>
#include <iostream>
#include <cuda_runtime.h>
#include "mg_descriptor.h"
#include "mg_utils.h"

namespace ML {

using namespace MLCommon;

void mean(float* mu, float* data, int D, int N, bool sample, bool rowMajor) {
	Stats::mean(mu, data, D, N, sample, rowMajor);
}

void mean(double* mu, double* data, int D, int N, bool sample, bool rowMajor) {
	Stats::mean(mu, data, D, N, sample, rowMajor);
}

void meanMG(MGDescriptorFloat *mu, MGDescriptorFloat *data, int n_gpus, bool sample, bool rowMajor, bool row_split) {

	TypeMG<float> *mus = new TypeMG<float>[n_gpus];
	TypeMG<float> *ins = new TypeMG<float>[n_gpus];

	for(int i = 0; i < n_gpus; i++) {

		cudaPointerAttributes att;
	    cudaError_t err = cudaPointerGetAttributes(&att, data[i].data);

	    TypeMG<float> *new_mg = new TypeMG<float>();
	    new_mg->d_data = data[i].data;
	    new_mg->gpu_id = att.device;
	    new_mg->n_cols = data[i].n_cols;
	    new_mg->n_rows = data[i].n_rows;
	    new_mg->stream = 0;

	    ins[i] = *new_mg;
	}

	for(int i = 0; i < n_gpus; i++) {

		cudaPointerAttributes att;
	    cudaError_t err = cudaPointerGetAttributes(&att, mu[i].data);

	    std::cout << "device: " << att.device << std::endl;
	    std::cout << "err: " << err << std::endl;

	    TypeMG<float> *new_mg = new TypeMG<float>();
	    new_mg->d_data = mu[i].data;
	    new_mg->gpu_id = att.device;
	    new_mg->n_cols = mu[i].n_cols;
	    new_mg->n_rows = mu[i].n_rows;
	    new_mg->stream = 0;

	    mus[i] = *new_mg;
	}

	Stats::meanMG(mus, ins, n_gpus, sample, rowMajor);
}


void meanMG(MGDescriptorDouble *mu, MGDescriptorDouble *data, int n_gpus, bool sample, bool rowMajor, bool row_split) {
	TypeMG<double> *mus = new TypeMG<double>[n_gpus];
	TypeMG<double> *ins = new TypeMG<double>[n_gpus];

	for(int i = 0; i < n_gpus; i++) {

		cudaPointerAttributes att;
	    cudaError_t err = cudaPointerGetAttributes(&att, data[i].data);

	    TypeMG<double> *new_mg = new TypeMG<double>();
	    new_mg->d_data = data[i].data;
	    new_mg->gpu_id = att.device;
	    new_mg->n_cols = data[i].n_cols;
	    new_mg->n_rows = data[i].n_rows;
	    new_mg->stream = 0;

	    ins[i] = *new_mg;
	}

	for(int i = 0; i < n_gpus; i++) {

		cudaPointerAttributes att;
	    cudaError_t err = cudaPointerGetAttributes(&att, mu[i].data);


	    TypeMG<double> *new_mg = new TypeMG<double>();
	    new_mg->d_data = mu[i].data;
	    new_mg->gpu_id = att.device;
	    new_mg->n_cols = mu[i].n_cols;
	    new_mg->n_rows = mu[i].n_rows;
	    new_mg->stream = 0;

	    mus[i] = *new_mg;
	}

	Stats::meanMG(mus, ins, n_gpus, sample, rowMajor);
}


/** @} */

}
;
// end namespace ML

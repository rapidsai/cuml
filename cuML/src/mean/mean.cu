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
#include <cuda_runtime.h>
#include "mg_descriptor.h"
#include "mg_utils.h"
#include "ml_utils.h"

namespace ML {

using namespace MLCommon;


void mean(float* mu, float* data, int D, int N, bool sample, bool rowMajor) {
	Stats::mean(mu, data, D, N, sample, rowMajor);
}

void mean(double* mu, double* data, int D, int N, bool sample, bool rowMajor) {
	Stats::mean(mu, data, D, N, sample, rowMajor);
}

void meanMG(MGDescriptorFloat *mu, MGDescriptorFloat *data, int n_gpus, bool sample, bool rowMajor, bool row_split) {

	TypeMG<float> *mus = convert_desc_to_typemg<float>(mu, n_gpus);
	TypeMG<float> *ins = convert_desc_to_typemg<float>(data, n_gpus);

	Stats::meanMG(mus, ins, n_gpus, sample, rowMajor);
}


void meanMG(MGDescriptorDouble *mu, MGDescriptorDouble *data, int n_gpus, bool sample, bool rowMajor, bool row_split) {

	TypeMG<double> *mus = convert_desc_to_typemg<double>(mu, n_gpus);
	TypeMG<double> *ins = convert_desc_to_typemg<double>(data, n_gpus);

	Stats::meanMG(mus, ins, n_gpus, sample, rowMajor);
}




/** @} */

}
;
// end namespace ML

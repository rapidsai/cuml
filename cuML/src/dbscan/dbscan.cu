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


#include "runner.h"
#include "dbscan_c.h"
#include "dbscan.h"

namespace ML {

using namespace Dbscan;


void dbscanFit(float *input, int n_rows, int n_cols, float eps, int min_pts,
		       int *labels) {
	dbscanFitImpl(input, n_rows, n_cols, eps, min_pts, labels);
}

void dbscanFit(double *input, int n_rows, int n_cols, double eps, int min_pts,
		       int *labels) {
	dbscanFitImpl(input, n_rows, n_cols, eps, min_pts, labels);
}

/** @} */

};
// end namespace ML

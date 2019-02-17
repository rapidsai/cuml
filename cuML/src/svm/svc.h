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

#include "ml_utils.h"
#include <cuda_utils.h>
#include <cub/cub.cuh>
#include <limits.h>
#include <linalg/ternary_op.h>

#include <linalg/gemv.h>
#include <stats/mean.h>
#include <stats/mean_center.h>
#include <linalg/add.h>
#include <linalg/subtract.h>
#include <linalg/norm.h>
#include <linalg/eltwise.h>
#include <linalg/unary_op.h>
#include <linalg/cublas_wrappers.h>

#include <linalg/map_then_reduce.h>

#include <iostream>

namespace ML {
namespace SVM {

using namespace MLCommon;


template <typename math_t>
__global__ void init_alpha_f(int* y, int n_rows, math_t* alpha, math_t* f) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx <n_rows) {
    	alpha[idx] = 0;
    	f[idx] = -y[idx];
    }
}

// determines weather the coefficient value a is in the upper set
template<typename math_t> DI bool in_upper(math_t a, math_t y, math_t C) {
  // return (0 < a && a < C) || ((y - 1) < eps && a < eps) || ((y + 1) < eps && (a - C) < eps);
  // since a is always clipped to lie in the [0 C] region, therefore this is equivalent with
  return (y < 0 && a > 0) || (y > 0 && a < C);
}
// determines weather the coefficient value a is in the lower set
template<typename math_t> DI bool in_lower(math_t a, math_t y, math_t C) {
  // return (0 < a && a < C) || ((y - 1) < eps && a < eps) || ((y + 1) < eps && (a - C) < eps);
  // since a is always clipped to lie in the [0 C] region, therefore this is equivalent with
  return (y < 0 && a < C) || (y > 0 && a > 0);
}

template<typename math_t>
void svcFit(math_t *input,
		    int n_rows,
		    int n_cols,
		    math_t *labels,
		    math_t *coef,
		    math_t C,
		    math_t tol,
		    cublasHandle_t cublas_handle,
		    cusolverDnHandle_t cusolver_handle) {

	ASSERT(n_cols > 1,
			"Parameter n_cols: number of columns cannot be less than two");
	ASSERT(n_rows > 1,
			"Parameter n_rows: number of rows cannot be less than two");

    std::cout<<"Hello SVM traning World!\n";

	math_t *alpha, *f, *tmp;

    allocate(alpha, n_rows);
	allocate(f, n_rows);
    allocate(tmp, n_rows);
    int TBP = 256

	init_alpha_f<<<ceildiv(n_rows,TPB),TPB>>>(labels, n_rows, alpha, f);

    // filter f for X_upper
    // Determine temporary device storage requirements
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, f, &u, 1);
    // Allocate temporary storage
    allocate(d_temp_storage, temp_storage_bytes);


    int n_iter = 0

    while (n_iter < 1000) {
    	int u;
    	//mapThenArgminReduce(u, n_rows, map, 0, f, );
        
        ternaryOp(tmp, f, alpha, y, int n_rows, 
                   [C]__device__(math_t f, math_t a, math_t y){ return in_upper(a, y, C) ? f : std::numeric_limits<math_t>::max(); }
                 );
    	// mask F values outside of X_upper
  	    cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, tmp, &u, 1);

  	    //gpusvm simply selects based on f value
        ternaryOp(tmp, f, alpha, y, int n_rows,
   	                     [C]__device__(math_t f, math_t a, math_t y){ return in_lower(a, y, C) ? f : std::numeric_limits<math_t>::min(); }
  	                   );
  	    /*
  	    // ThunderSVM : l_formula needs to be implemented (f_u - f_i)^2/eta_i
        ternaryOp(tmp, f, alpha, y, n_rows, 
                   [C, u, f]__device__(math_t fval, math_t a, math_t y) { 
                       return in_lower(a, y, c) && f[u] < fval ? l_formula(fval, f[u], eta) : std::numeric_limits<math_t>::max(); }
                 );
        */

        int l;
        cub::DeviceReduce::ArgMax(d_temp_storage, temp_storage_bytes, tmp, &l, 1);
        
        math_t alpha_l_new = alpha[l] + y[l] * (f[u] - f[l]) / eta(u,l);
        if (alpha_l_new < 0) alpha_l_new = 0;
        else if (alpha_l_new > c) alpha_l_new = C;
        Dl = (alpha_l_new - alpha[l]) * y[l];
        
        alpha_u_new = alpha[u] - y[l]*y[u]*alpha_l_diff;
        if (alpha_u_new < 0) alpha_u_new = 0;
        else if (alpha_u_new > c) alpha_u_new = C;
        Du = (alpha_u_new - alpha[u]) * y[u];
        
        // update coefficients
        ternaryOp(f, f, Ku, Kl, n_rows, 
                   [C, u, Dalpha_l, Dalpha_u]__device__(math_t f, math_t Ku, math_t Kl) { return + Du * Ku + Dl* Kl}
                 );
        math_t fmax;
        cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, f, &fmax, 1); 
        if (f[l] > f[u] + 2*tol) { // error, f[u] should not be evaluated on the host
            break; 
        }
    }


	CUDA_CHECK(cudaFree(alpha));
	CUDA_CHECK(cudaFree(f));
    CUDA_CHECK(cudaFree(tmp));
}
/*
template<typename math_t>
void svcPredict(const math_t *input, int n_rows, int n_cols, const math_t *coef,
		math_t intercept, math_t *preds, ML::loss_funct loss, cublasHandle_t cublas_handle) {

	ASSERT(n_cols > 1,
			"Parameter n_cols: number of columns cannot be less than two");
	ASSERT(n_rows > 1,
			"Parameter n_rows: number of rows cannot be less than two");

    std::cout<<"Hello SVM prediction World!\n";
}
*/
/** @} */
}
;
}
;
// end namespace ML

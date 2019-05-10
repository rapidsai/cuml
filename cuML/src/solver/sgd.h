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

#include "ml_utils.h"
#include <cuda_utils.h>
#include <linalg/gemv.h>
#include <stats/mean.h>
#include <stats/mean_center.h>
#include <linalg/add.h>
#include <linalg/subtract.h>
#include <linalg/norm.h>
#include <linalg/eltwise.h>
#include <linalg/unary_op.h>
#include <linalg/cublas_wrappers.h>
#include <matrix/math.h>
#include <matrix/matrix.h>
#include "glm/preprocess.h"
#include "shuffle.h"
#include <functions/linearReg.h>
#include <functions/logisticReg.h>
#include <functions/hinge.h>
#include "learning_rate.h"

namespace ML {
namespace Solver {

using namespace MLCommon;

template<typename math_t>
void sgdFit(math_t *input,
		    int n_rows,
		    int n_cols,
		    math_t *labels,
		    math_t *coef,
		    math_t *intercept,
		    bool fit_intercept,
		    int batch_size,
		    int epochs,
		    ML::lr_type lr_type,
		    math_t eta0,
		    math_t power_t,
		    ML::loss_funct loss,
		    Functions::penalty penalty,
		    math_t alpha,
		    math_t l1_ratio,
		    bool shuffle,
		    math_t tol,
		    int n_iter_no_change,
		    cublasHandle_t cublas_handle,
		    cusolverDnHandle_t cusolver_handle,
			cudaStream_t stream) {

	ASSERT(n_cols > 0,
			"Parameter n_cols: number of columns cannot be less than one");
	ASSERT(n_rows > 1,
			"Parameter n_rows: number of rows cannot be less than two");

	math_t *mu_input = NULL;
	math_t *mu_labels = NULL;
	math_t *norm2_input = NULL;

	if (fit_intercept) {
		allocate(mu_input, n_cols);
		allocate(mu_labels, 1);

		GLM::preProcessData(input, n_rows, n_cols, labels, intercept, mu_input,
				mu_labels, norm2_input, fit_intercept, false, cublas_handle,
				cusolver_handle, stream);
	}

	math_t *grads = NULL;
	math_t *input_batch = NULL;
	math_t *labels_batch = NULL;
	math_t *loss_value = NULL;
	int *indices = NULL;

	allocate(grads, n_cols, true);
	allocate(indices, batch_size);
	allocate(input_batch, batch_size * n_cols);
	allocate(labels_batch, batch_size);
	allocate(loss_value, 1);

	math_t prev_loss_value = math_t(0);
	math_t curr_loss_value = math_t(0);

	std::vector<int> rand_indices(n_rows);
	std::mt19937 g(rand());
	initShuffle(rand_indices, g);

	math_t t = math_t(1);
	math_t learning_rate = math_t(0);
	if (lr_type == ML::lr_type::ADAPTIVE) {
		learning_rate = eta0;
	} else if (lr_type == ML::lr_type::OPTIMAL) {
		eta0 = calOptimalInit(alpha);
	}

	int n_iter_no_change_curr = 0;

	for (int i = 0; i < epochs; i++) {
		int cbs = 0;
		int j = 0;

		if (i > 0 && shuffle) {
			Solver::shuffle(rand_indices, g);
		}

		while (j < n_rows) {
			if ((j + batch_size) > n_rows) {
				cbs = n_rows - j;
			} else {
				cbs = batch_size;
			}

			if (cbs == 0)
				break;

			updateDevice(indices, &rand_indices[j], cbs, stream);
			Matrix::copyRows(input, n_rows, n_cols, input_batch, indices, cbs, stream);
			Matrix::copyRows(labels, n_rows, 1, labels_batch, indices, cbs, stream);

			if (loss == ML::loss_funct::SQRD_LOSS) {
				Functions::linearRegLossGrads(input_batch, cbs, n_cols, labels_batch,
						coef, grads, penalty, alpha, l1_ratio, cublas_handle, stream);
			} else if (loss == ML::loss_funct::LOG) {
				Functions::logisticRegLossGrads(input_batch, cbs, n_cols, labels_batch,
										coef, grads, penalty, alpha, l1_ratio, cublas_handle, stream);
			} else if (loss == ML::loss_funct::HINGE) {
				Functions::hingeLossGrads(input_batch, cbs, n_cols, labels_batch,
														coef, grads, penalty, alpha, l1_ratio, cublas_handle, stream);
			} else {
				ASSERT(false,
						"sgd.h: Other loss functions have not been implemented yet!");
			}

			if (lr_type != ML::lr_type::ADAPTIVE)
			    learning_rate = calLearningRate(lr_type, eta0, power_t, alpha, t);

			LinAlg::scalarMultiply(grads, grads, learning_rate, n_cols, stream);
			LinAlg::subtract(coef, coef, grads, n_cols, stream);

			j = j + cbs;
			t = t + 1;
		}

		if (tol > math_t(0)) {
			if (loss == ML::loss_funct::SQRD_LOSS) {
			    Functions::linearRegLoss(input, n_rows, n_cols, labels, coef, loss_value,
					    penalty, alpha, l1_ratio, cublas_handle, stream);
			} else if (loss == ML::loss_funct::LOG) {
				Functions::logisticRegLoss(input, n_rows, n_cols, labels, coef, loss_value,
						penalty, alpha, l1_ratio, cublas_handle, stream);
			} else if (loss == ML::loss_funct::HINGE) {
				Functions::hingeLoss(input, n_rows, n_cols, labels, coef, loss_value,
						penalty, alpha, l1_ratio, cublas_handle, stream);
			}

			updateHost(&curr_loss_value, loss_value, 1, stream);
                        CUDA_CHECK(cudaStreamSynchronize(stream));

			if (i > 0) {
                if (curr_loss_value > (prev_loss_value - tol)) {
                	n_iter_no_change_curr = n_iter_no_change_curr + 1;
                	if (n_iter_no_change_curr > n_iter_no_change) {
                		if (lr_type == ML::lr_type::ADAPTIVE && learning_rate > math_t(1e-6)) {
                			learning_rate = learning_rate / math_t(5);
                			n_iter_no_change_curr = 0;
                		} else {
                		    break;
                		}
                	}
                } else {
                	n_iter_no_change_curr = 0;
                }
			}

			prev_loss_value = curr_loss_value;
		}
	}

	if (grads != NULL)
	    CUDA_CHECK(cudaFree(grads));
	if (indices != NULL)
	    CUDA_CHECK(cudaFree(indices));
	if (input_batch != NULL)
	    CUDA_CHECK(cudaFree(input_batch));
	if (labels_batch != NULL)
	    CUDA_CHECK(cudaFree(labels_batch));
	if (loss_value != NULL)
	    CUDA_CHECK(cudaFree(loss_value));

	if (fit_intercept) {
		GLM::postProcessData(input, n_rows, n_cols, labels, coef, intercept,
				mu_input, mu_labels, norm2_input, fit_intercept, false,
				cublas_handle, cusolver_handle, stream);

		if (mu_input != NULL)
			CUDA_CHECK(cudaFree(mu_input));
		if (mu_labels != NULL)
			CUDA_CHECK(cudaFree(mu_labels));
	} else {
		*intercept = math_t(0);
	}

}

template<typename math_t>
void sgdPredict(const math_t *input, int n_rows, int n_cols, const math_t *coef,
		math_t intercept, math_t *preds, ML::loss_funct loss, cublasHandle_t cublas_handle,
		cudaStream_t stream) {

	ASSERT(n_cols > 0,
			"Parameter n_cols: number of columns cannot be less than one");
	ASSERT(n_rows > 1,
			"Parameter n_rows: number of rows cannot be less than two");

	if (loss == ML::loss_funct::SQRD_LOSS) {
		Functions::linearRegH(input, n_rows, n_cols, coef, preds, intercept, cublas_handle, stream);
	} else if (loss == ML::loss_funct::LOG) {
		Functions::logisticRegH(input, n_rows, n_cols, coef, preds, intercept, cublas_handle, stream);
	} else if (loss == ML::loss_funct::HINGE) {
		Functions::hingeH(input, n_rows, n_cols, coef, preds, intercept, cublas_handle, stream);
	}
}

template<typename math_t>
void sgdPredictBinaryClass(const math_t *input, int n_rows, int n_cols, const math_t *coef,
		math_t intercept, math_t *preds, ML::loss_funct loss, cublasHandle_t cublas_handle, cudaStream_t stream) {

	sgdPredict(input, n_rows, n_cols, coef, intercept, preds, loss, cublas_handle, stream);

	math_t scalar = math_t(1);
	if (loss == ML::loss_funct::SQRD_LOSS || loss == ML::loss_funct::LOG) {
		LinAlg::unaryOp(preds, preds, n_rows, [scalar] __device__ (math_t in) {
		                                                  	  if (in >= math_t(0.5))
		                                                  		  return math_t(1);
		                                                  	  else
		                                                  		  return math_t(0);
                                                        },
                                                        stream);
	} else if (loss == ML::loss_funct::HINGE) {
		LinAlg::unaryOp(preds, preds, n_rows, [scalar] __device__ (math_t in) {
				                                              if (in >= math_t(0.0))
				                                                  return math_t(1);
				                                              else
				                                                  return math_t(0);
			                                                  },
                                                        stream);
	}

}

/** @} */
}
;
}
;
// end namespace ML

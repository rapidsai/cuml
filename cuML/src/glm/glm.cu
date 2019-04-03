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
#include "ols.h"
#include "ridge.h"
#include "glm_c.h"
#include "cuML.hpp"

namespace ML {
namespace GLM {

using namespace MLCommon;

void olsFit(float *input, int n_rows, int n_cols, float *labels, float *coef,
            float *intercept, bool fit_intercept, bool normalize, int algo) {
    ///@todo: expose cumlHandle in the interface, then remove the construction of
    /// handle and the stream objects below
    cumlHandle handle;
    cudaStream_t s_;
    CUDA_CHECK(cudaStreamCreate(&s_));
    handle.setStream(s_);

    olsFit(handle.getImpl(), input, n_rows, n_cols, labels, coef, intercept, fit_intercept,
           normalize, algo);

    ///@todo: remove this after updating this method to expose cumlHandle
    CUDA_CHECK(cudaStreamSynchronize(s_));
    CUDA_CHECK(cudaStreamDestroy(s_));
}

void olsFit(double *input, int n_rows, int n_cols, double *labels, double *coef,
            double *intercept, bool fit_intercept, bool normalize, int algo) {
    ///@todo: expose cumlHandle in the interface, then remove the construction of
    /// handle and the stream objects below
    cumlHandle handle;
    cudaStream_t s_;
    CUDA_CHECK(cudaStreamCreate(&s_));
    handle.setStream(s_);

    olsFit(handle.getImpl(), input, n_rows, n_cols, labels, coef, intercept, fit_intercept,
           normalize, algo);

    ///@todo: remove this after updating this method to expose cumlHandle
    CUDA_CHECK(cudaStreamSynchronize(s_));
    CUDA_CHECK(cudaStreamDestroy(s_));
}

void olsPredict(const float *input, int n_rows, int n_cols, const float *coef,
		float intercept, float *preds) {
    ///@todo: expose cumlHandle in the interface, then remove the construction of
    /// handle and the stream objects below
    cumlHandle handle;
    cudaStream_t s_;
    CUDA_CHECK(cudaStreamCreate(&s_));
    handle.setStream(s_);

    olsPredict(handle.getImpl(), input, n_rows, n_cols, coef, intercept, preds);

    ///@todo: remove this after updating this method to expose cumlHandle
    CUDA_CHECK(cudaStreamSynchronize(s_));
    CUDA_CHECK(cudaStreamDestroy(s_));
}

void olsPredict(const double *input, int n_rows, int n_cols, const double *coef,
		double intercept, double *preds) {
    ///@todo: expose cumlHandle in the interface, then remove the construction of
    /// handle and the stream objects below
    cumlHandle handle;
    cudaStream_t s_;
    CUDA_CHECK(cudaStreamCreate(&s_));
    handle.setStream(s_);

    olsPredict(handle.getImpl(), input, n_rows, n_cols, coef, intercept, preds);

    ///@todo: remove this after updating this method to expose cumlHandle
    CUDA_CHECK(cudaStreamSynchronize(s_));
    CUDA_CHECK(cudaStreamDestroy(s_));
}

void ridgeFit(float *input, int n_rows, int n_cols, float *labels, float *alpha,
		int n_alpha, float *coef, float *intercept, bool fit_intercept,
		bool normalize, int algo) {
    ///@todo: expose cumlHandle in the interface, then remove the construction of
    /// handle and the stream objects below
    cumlHandle handle;
    cudaStream_t s_;
    CUDA_CHECK(cudaStreamCreate(&s_));
    handle.setStream(s_);

    ridgeFit(handle.getImpl(), input, n_rows, n_cols, labels, alpha, n_alpha, coef, intercept,
             fit_intercept, normalize, algo);

    ///@todo: remove this after updating this method to expose cumlHandle
    CUDA_CHECK(cudaStreamSynchronize(s_));
    CUDA_CHECK(cudaStreamDestroy(s_));
}

void ridgeFit(double *input, int n_rows, int n_cols, double *labels,
		double *alpha, int n_alpha, double *coef, double *intercept,
		bool fit_intercept, bool normalize, int algo) {
    ///@todo: expose cumlHandle in the interface, then remove the construction of
    /// handle and the stream objects below
    cumlHandle handle;
    cudaStream_t s_;
    CUDA_CHECK(cudaStreamCreate(&s_));
    handle.setStream(s_);

    ridgeFit(handle.getImpl(), input, n_rows, n_cols, labels, alpha, n_alpha, coef, intercept,
             fit_intercept, normalize, algo);

    ///@todo: remove this after updating this method to expose cumlHandle
    CUDA_CHECK(cudaStreamSynchronize(s_));
    CUDA_CHECK(cudaStreamDestroy(s_));
}

void ridgePredict(const float *input, int n_rows, int n_cols, const float *coef,
		float intercept, float *preds) {
    ///@todo: expose cumlHandle in the interface, then remove the construction of
    /// handle and the stream objects below
    cumlHandle handle;
    cudaStream_t s_;
    CUDA_CHECK(cudaStreamCreate(&s_));
    handle.setStream(s_);

    ridgePredict(handle.getImpl(), input, n_rows, n_cols, coef, intercept, preds);

    ///@todo: remove this after updating this method to expose cumlHandle
    CUDA_CHECK(cudaStreamSynchronize(s_));
    CUDA_CHECK(cudaStreamDestroy(s_));
}

void ridgePredict(const double *input, int n_rows, int n_cols, const double *coef,
		double intercept, double *preds) {
    ///@todo: expose cumlHandle in the interface, then remove the construction of
    /// handle and the stream objects below
    cumlHandle handle;
    cudaStream_t s_;
    CUDA_CHECK(cudaStreamCreate(&s_));
    handle.setStream(s_);

    ridgePredict(handle.getImpl(), input, n_rows, n_cols, coef, intercept, preds);

    ///@todo: remove this after updating this method to expose cumlHandle
    CUDA_CHECK(cudaStreamSynchronize(s_));
    CUDA_CHECK(cudaStreamDestroy(s_));
}

}
}

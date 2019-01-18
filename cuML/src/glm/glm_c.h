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

namespace ML {
namespace GLM {

// OLS
void olsFit(float *input, int n_rows, int n_cols, float *labels, float *coef,
		float *intercept, bool fit_intercept, bool normalize, int algo = 0);
void olsFit(double *input, int n_rows, int n_cols, double *labels, double *coef,
		double *intercept, bool fit_intercept, bool normalize, int algo = 0);
void olsPredict(const float *input, int n_rows, int n_cols, const float *coef,
		float intercept, float *preds);
void olsPredict(const double *input, int n_rows, int n_cols, const double *coef,
		double intercept, double *preds);

// Ridge
void ridgeFit(float *input, int n_rows, int n_cols, float *labels, float *alpha,
		int n_alpha, float *coef, float *intercept, bool fit_intercept,
		bool normalize, int algo = 0);

void ridgeFit(double *input, int n_rows, int n_cols, double *labels,
		double *alpha, int n_alpha, double *coef, double *intercept,
		bool fit_intercept, bool normalize, int algo = 0);

void ridgePredict(const float *input, int n_rows, int n_cols, const float *coef,
		float intercept, float *preds);

void ridgePredict(const double *input, int n_rows, int n_cols,
		const double *coef, double intercept, double *preds);

}
}

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

namespace ML {
namespace GLM {


void olsFitSPMG(float *h_input, int n_rows, int n_cols, float *h_labels, float *h_coef,
        float *intercept, bool fit_intercept, bool normalize, int *gpu_ids, int n_gpus);
void olsFitSPMG(double *h_input, int n_rows, int n_cols, double *h_labels, double *h_coef,
        double *intercept, bool fit_intercept, bool normalize, int *gpu_ids, int n_gpus);
void olsPredictSPMG(float *input, int n_rows, int n_cols, float *h_coef,
        float intercept, float *preds, int *gpu_ids, int n_gpus);
void olsPredictSPMG(double *input, int n_rows, int n_cols, double *h_coef,
        double intercept, double *preds, int *gpu_ids, int n_gpus);

void spmgOlsFit(float **input, int *input_cols, int n_rows, int n_cols,
                float **labels, int *label_rows, float **coef, int *coef_cols,
                float *intercept, bool fit_intercept, bool normalize, int n_gpus);

void spmgOlsFit(double **input, int *input_cols, int n_rows, int n_cols,
                double **labels, int *label_rows, double **coef, int *coef_cols,
                double *intercept, bool fit_intercept, bool normalize, int n_gpus);

void spmgOlsPredict(float **input, int *input_cols, int n_rows, int n_cols,
                    float **coef, int *coef_cols, float intercept,
                    float **preds, int *pred_cols, int n_gpus);

void spmgOlsPredict(double **input, int *input_cols, int n_rows, int n_cols,
                    double **coef, int *coef_cols, double intercept,
                    double **preds, int *pred_cols, int n_gpus);

}
}

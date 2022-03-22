/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

#include "ols.cuh"
#include "qn/qn.cuh"
#include "ridge.cuh"
#include <cuml/linear_model/glm.hpp>

namespace raft {
class handle_t;
}

namespace ML {
namespace GLM {

void olsFit(const raft::handle_t& handle,
            float* input,
            int n_rows,
            int n_cols,
            float* labels,
            float* coef,
            float* intercept,
            bool fit_intercept,
            bool normalize,
            int algo,
            float* sample_weight)
{
  olsFit(handle,
         input,
         n_rows,
         n_cols,
         labels,
         coef,
         intercept,
         fit_intercept,
         normalize,
         handle.get_stream(),
         algo,
         sample_weight);
}

void olsFit(const raft::handle_t& handle,
            double* input,
            int n_rows,
            int n_cols,
            double* labels,
            double* coef,
            double* intercept,
            bool fit_intercept,
            bool normalize,
            int algo,
            double* sample_weight)
{
  olsFit(handle,
         input,
         n_rows,
         n_cols,
         labels,
         coef,
         intercept,
         fit_intercept,
         normalize,
         handle.get_stream(),
         algo,
         sample_weight);
}

void gemmPredict(const raft::handle_t& handle,
                 const float* input,
                 int n_rows,
                 int n_cols,
                 const float* coef,
                 float intercept,
                 float* preds)
{
  gemmPredict(handle, input, n_rows, n_cols, coef, intercept, preds, handle.get_stream());
}

void gemmPredict(const raft::handle_t& handle,
                 const double* input,
                 int n_rows,
                 int n_cols,
                 const double* coef,
                 double intercept,
                 double* preds)
{
  gemmPredict(handle, input, n_rows, n_cols, coef, intercept, preds, handle.get_stream());
}

void ridgeFit(const raft::handle_t& handle,
              float* input,
              int n_rows,
              int n_cols,
              float* labels,
              float* alpha,
              int n_alpha,
              float* coef,
              float* intercept,
              bool fit_intercept,
              bool normalize,
              int algo)
{
  ridgeFit(handle,
           input,
           n_rows,
           n_cols,
           labels,
           alpha,
           n_alpha,
           coef,
           intercept,
           fit_intercept,
           normalize,
           handle.get_stream(),
           algo);
}

void ridgeFit(const raft::handle_t& handle,
              double* input,
              int n_rows,
              int n_cols,
              double* labels,
              double* alpha,
              int n_alpha,
              double* coef,
              double* intercept,
              bool fit_intercept,
              bool normalize,
              int algo)
{
  ridgeFit(handle,
           input,
           n_rows,
           n_cols,
           labels,
           alpha,
           n_alpha,
           coef,
           intercept,
           fit_intercept,
           normalize,
           handle.get_stream(),
           algo);
}

template <typename T, typename I>
void qnFit(const raft::handle_t& cuml_handle,
           const qn_params& pams,
           T* X,
           bool X_col_major,
           T* y,
           I N,
           I D,
           I C,
           T* w0,
           T* f,
           int* num_iters,
           T* sample_weight)
{
  qnFit<T>(cuml_handle,
           pams,
           X,
           X_col_major,
           y,
           N,
           D,
           C,
           w0,
           f,
           num_iters,
           cuml_handle.get_stream(),
           sample_weight);
}

template void qnFit<float>(const raft::handle_t&,
                           const qn_params&,
                           float*,
                           bool,
                           float*,
                           int,
                           int,
                           int,
                           float*,
                           float*,
                           int*,
                           float*);
template void qnFit<double>(const raft::handle_t&,
                            const qn_params&,
                            double*,
                            bool,
                            double*,
                            int,
                            int,
                            int,
                            double*,
                            double*,
                            int*,
                            double*);

template <typename T, typename I>
void qnFitSparse(const raft::handle_t& cuml_handle,
                 const qn_params& pams,
                 T* X_values,
                 I* X_cols,
                 I* X_row_ids,
                 I X_nnz,
                 T* y,
                 I N,
                 I D,
                 I C,
                 T* w0,
                 T* f,
                 int* num_iters,
                 T* sample_weight)
{
  qnFitSparse<T>(cuml_handle,
                 pams,
                 X_values,
                 X_cols,
                 X_row_ids,
                 X_nnz,
                 y,
                 N,
                 D,
                 C,
                 w0,
                 f,
                 num_iters,
                 cuml_handle.get_stream(),
                 sample_weight);
}

template void qnFitSparse<float>(const raft::handle_t&,
                                 const qn_params&,
                                 float*,
                                 int*,
                                 int*,
                                 int,
                                 float*,
                                 int,
                                 int,
                                 int,
                                 float*,
                                 float*,
                                 int*,
                                 float*);
template void qnFitSparse<double>(const raft::handle_t&,
                                  const qn_params&,
                                  double*,
                                  int*,
                                  int*,
                                  int,
                                  double*,
                                  int,
                                  int,
                                  int,
                                  double*,
                                  double*,
                                  int*,
                                  double*);

template <typename T, typename I>
void qnDecisionFunction(const raft::handle_t& cuml_handle,
                        const qn_params& pams,
                        T* X,
                        bool X_col_major,
                        I N,
                        I D,
                        I C,
                        T* params,
                        T* scores)
{
  qnDecisionFunction<T>(
    cuml_handle, pams, X, X_col_major, N, D, C, params, scores, cuml_handle.get_stream());
}

template void qnDecisionFunction<float>(
  const raft::handle_t&, const qn_params&, float*, bool, int, int, int, float*, float*);
template void qnDecisionFunction<double>(
  const raft::handle_t&, const qn_params&, double*, bool, int, int, int, double*, double*);

template <typename T, typename I>
void qnDecisionFunctionSparse(const raft::handle_t& cuml_handle,
                              const qn_params& pams,
                              T* X_values,
                              I* X_cols,
                              I* X_row_ids,
                              I X_nnz,
                              I N,
                              I D,
                              I C,
                              T* params,
                              T* scores)
{
  qnDecisionFunctionSparse<T>(cuml_handle,
                              pams,
                              X_values,
                              X_cols,
                              X_row_ids,
                              X_nnz,
                              N,
                              D,
                              C,
                              params,
                              scores,
                              cuml_handle.get_stream());
}

template void qnDecisionFunctionSparse<float>(
  const raft::handle_t&, const qn_params&, float*, int*, int*, int, int, int, int, float*, float*);
template void qnDecisionFunctionSparse<double>(const raft::handle_t&,
                                               const qn_params&,
                                               double*,
                                               int*,
                                               int*,
                                               int,
                                               int,
                                               int,
                                               int,
                                               double*,
                                               double*);

template <typename T, typename I>
void qnPredict(const raft::handle_t& cuml_handle,
               const qn_params& pams,
               T* X,
               bool X_col_major,
               I N,
               I D,
               I C,
               T* params,
               T* scores)
{
  qnPredict<T>(
    cuml_handle, pams, X, X_col_major, N, D, C, params, scores, cuml_handle.get_stream());
}

template void qnPredict<float>(
  const raft::handle_t&, const qn_params&, float*, bool, int, int, int, float*, float*);
template void qnPredict<double>(
  const raft::handle_t&, const qn_params&, double*, bool, int, int, int, double*, double*);

template <typename T, typename I>
void qnPredictSparse(const raft::handle_t& cuml_handle,
                     const qn_params& pams,
                     T* X_values,
                     I* X_cols,
                     I* X_row_ids,
                     I X_nnz,
                     I N,
                     I D,
                     I C,
                     T* params,
                     T* preds)
{
  qnPredictSparse<T>(cuml_handle,
                     pams,
                     X_values,
                     X_cols,
                     X_row_ids,
                     X_nnz,
                     N,
                     D,
                     C,
                     params,
                     preds,
                     cuml_handle.get_stream());
}

template void qnPredictSparse<float>(
  const raft::handle_t&, const qn_params&, float*, int*, int*, int, int, int, int, float*, float*);
template void qnPredictSparse<double>(const raft::handle_t&,
                                      const qn_params&,
                                      double*,
                                      int*,
                                      int*,
                                      int,
                                      int,
                                      int,
                                      int,
                                      double*,
                                      double*);

}  // namespace GLM
}  // namespace ML

# Copyright (c) 2018, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from libcpp cimport bool

cdef extern from "solver/solver_c.h" namespace "ML::Solver":

    cdef void sgdFit(float *input,
	                 int n_rows,
	                 int n_cols,
	                 float *labels,
	                 float *coef,
	                 float *intercept,
	                 bool fit_intercept,
	                 int batch_size,
	                 int epochs,
	                 int lr_type,
	                 float eta0,
	                 float power_t,
	                 int loss,
	                 int penalty,
	                 float alpha,
	                 float l1_ratio,
	                 bool shuffle,
	                 float tol,
	                 int n_iter_no_change)

    
    cdef void sgdFit(double *input,
	                 int n_rows,
	                 int n_cols,
	                 double *labels,
	                 double *coef,
	                 double *intercept,
	                 bool fit_intercept,
	                 int batch_size,
	                 int epochs,
	                 int lr_type,
	                 double eta0,
	                 double power_t,
	                 int loss,
	                 int penalty,
	                 double alpha,
	                 double l1_ratio,
	                 bool shuffle,
	                 double tol,
	                 int n_iter_no_change)
	                 
    cdef void sgdPredict(const float *input, 
                         int n_rows, 
                         int n_cols, 
                         const float *coef,
                         float intercept, 
                         float *preds,
                         int loss)

    cdef void sgdPredict(const double *input, 
                         int n_rows, 
                         int n_cols,
                         const double *coef, 
                         double intercept, 
                         double *preds,
                         int loss)
                         
    cdef void sgdPredictBinaryClass(const float *input, 
                         int n_rows, 
                         int n_cols, 
                         const float *coef,
                         float intercept, 
                         float *preds,
                         int loss)

    cdef void sgdPredictBinaryClass(const double *input, 
                         int n_rows, 
                         int n_cols,
                         const double *coef, 
                         double intercept, 
                         double *preds,
                         int loss)
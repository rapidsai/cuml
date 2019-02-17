# Copyright (c) 2019, NVIDIA CORPORATION.
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

cdef extern from "svm/svm_c.h" namespace "ML::SVM":

    cdef void svcFit(float *input,
	                 int n_rows,
	                 int n_cols,
	                 float *labels,
	                 float *coef,
	                 float C,
	                 float tol)

    
    cdef void svcFit(double *input,
	                 int n_rows,
	                 int n_cols,
	                 double *labels,
	                 double *coef,
	                 double C,
	                 double tol)
	                 

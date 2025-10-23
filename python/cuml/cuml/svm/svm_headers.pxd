#
# Copyright (c) 2025, NVIDIA CORPORATION.
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
from pylibraft.common.handle cimport handle_t

from cuml.internals.logger cimport level_enum


cdef extern from "cuml/matrix/kernel_params.hpp" namespace "ML::matrix" nogil:

    enum class KernelType:
        LINEAR, POLYNOMIAL, RBF, TANH

    cdef struct KernelParams:
        KernelType kernel
        int degree
        double gamma
        double coef0


cdef extern from "cuml/svm/svm_parameter.h" namespace "ML::SVM" nogil:

    enum SvmType:
        C_SVC,
        NU_SVC,
        EPSILON_SVR,
        NU_SVR

    cdef struct SvmParameter:
        double C
        double cache_size
        int max_iter
        int nochange_steps
        double tol
        level_enum verbosity
        double epsilon
        SvmType svmType


cdef extern from "cuml/svm/svm_model.h" namespace "ML::SVM" nogil:

    cdef cppclass SupportStorage[math_t]:
        int nnz
        int* indptr
        int* indices
        math_t* data

    cdef cppclass SvmModel[math_t]:
        int n_support
        int n_cols
        math_t b
        math_t *dual_coefs
        SupportStorage[math_t] support_matrix
        int *support_idx
        int n_classes
        math_t *unique_labels


cdef extern from "cuml/svm/svc.hpp" namespace "ML::SVM" nogil:

    cdef void svcPredict[math_t](
        const handle_t &handle,
        math_t* data,
        int n_rows,
        int n_cols,
        KernelParams &kernel_params,
        const SvmModel[math_t] &model,
        math_t *preds,
        math_t buffer_size,
        bool predict_class,
    ) except +

    cdef void svcPredictSparse[math_t](
        const handle_t &handle,
        int* indptr,
        int* indices,
        math_t* data,
        int n_rows,
        int n_cols,
        int nnz,
        KernelParams &kernel_params,
        const SvmModel[math_t] &model,
        math_t *preds,
        math_t buffer_size,
        bool predict_class,
    ) except +

    cdef void svmFreeBuffers[math_t](
        const handle_t &handle,
        SvmModel[math_t] &m,
    ) except +


cdef extern from "cuml/svm/svr.hpp" namespace "ML::SVM" nogil:

    cdef void svrFit[math_t](
        const handle_t &handle,
        math_t* data,
        int n_rows,
        int n_cols,
        math_t *y,
        const SvmParameter &param,
        KernelParams &kernel_params,
        SvmModel[math_t] &model,
        const math_t *sample_weight,
    ) except+

    cdef void svrFitSparse[math_t](
        const handle_t &handle,
        int* indptr,
        int* indices,
        math_t* data,
        int n_rows,
        int n_cols,
        int nnz,
        math_t *y,
        const SvmParameter &param,
        KernelParams &kernel_params,
        SvmModel[math_t] &model,
        const math_t *sample_weight,
    ) except+

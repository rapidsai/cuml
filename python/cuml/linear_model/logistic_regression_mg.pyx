#
# Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
# distutils: language = c++

from cuml.internals.safe_imports import gpu_only_import
cp = gpu_only_import('cupy')
from cuml.internals.safe_imports import cpu_only_import
np = cpu_only_import('numpy')

from libcpp cimport bool
from libc.stdint cimport uintptr_t

from cuml.common import input_to_cuml_array
import numpy as np

from cuml.internals.array import CumlArray
from cuml.internals.api_decorators import device_interop_preparation
from cuml.linear_model import LogisticRegression 
from cuml.solvers.qn import QN
from cuml.solvers.qn import QNParams

from pylibraft.common.handle cimport handle_t

from cuml.solvers.qn import __is_col_major

# the cdef was copied from cuml.linear_model.qn
cdef extern from "cuml/linear_model/glm.hpp" namespace "ML::GLM" nogil:

    cdef enum qn_loss_type "ML::GLM::qn_loss_type":
        QN_LOSS_LOGISTIC "ML::GLM::QN_LOSS_LOGISTIC"
        QN_LOSS_SQUARED  "ML::GLM::QN_LOSS_SQUARED"
        QN_LOSS_SOFTMAX  "ML::GLM::QN_LOSS_SOFTMAX"
        QN_LOSS_SVC_L1   "ML::GLM::QN_LOSS_SVC_L1"
        QN_LOSS_SVC_L2   "ML::GLM::QN_LOSS_SVC_L2"
        QN_LOSS_SVR_L1   "ML::GLM::QN_LOSS_SVR_L1"
        QN_LOSS_SVR_L2   "ML::GLM::QN_LOSS_SVR_L2"
        QN_LOSS_ABS      "ML::GLM::QN_LOSS_ABS"
        QN_LOSS_UNKNOWN  "ML::GLM::QN_LOSS_UNKNOWN"

    cdef struct qn_params:
        qn_loss_type loss
        double penalty_l1
        double penalty_l2
        double grad_tol
        double change_tol
        int max_iter
        int linesearch_max_iter
        int lbfgs_memory
        int verbose
        bool fit_intercept
        bool penalty_normalized


cdef extern from "cuml/linear_model/qn_mg.hpp" namespace "ML::GLM::opg" nogil:

    void qnFit(
        const handle_t& handle,
        const qn_params& pams, 
        float *X,
        bool X_col_major,
        float *y,
        int N,
        int D,
        int C,
        float *w0,
        float *f,
        int *num_iters,
        int n_samples,
        int rank,
        int n_ranks) except +


class LogisticRegressionMG(LogisticRegression):
    
    def __init__(self, *, handle=None):
        super().__init__(handle=handle)

    def prepare_for_fit(self, n_classes):
        self.qnparams = QNParams(
            loss=self.loss,
            penalty_l1=self.l1_strength,
            penalty_l2=self.l2_strength,
            grad_tol=self.tol,
            change_tol=self.delta
            if self.delta is not None else (self.tol * 0.01),
            max_iter=self.max_iter,
            linesearch_max_iter=self.linesearch_max_iter,
            lbfgs_memory=self.lbfgs_memory,
            verbose=self.verbose,
            fit_intercept=self.fit_intercept,
            penalty_normalized=self.penalty_normalized
        )

        # modified
        qnpams = self.qnparams.params

        # modified qnp
        solves_classification = qnpams['loss'] in {
            qn_loss_type.QN_LOSS_LOGISTIC,
            qn_loss_type.QN_LOSS_SOFTMAX,
            qn_loss_type.QN_LOSS_SVC_L1,
            qn_loss_type.QN_LOSS_SVC_L2
        }
        solves_multiclass = qnpams['loss'] in {
            qn_loss_type.QN_LOSS_SOFTMAX
        }

        if solves_classification:
            self._num_classes = n_classes
        else:
            self._num_classes = 1

        if not solves_multiclass and self._num_classes > 2:
            raise ValueError(
                f"The selected solver ({self.loss}) does not support"
                f" more than 2 classes ({self._num_classes} discovered).")

        if qnpams['loss'] == qn_loss_type.QN_LOSS_SOFTMAX \
           and self._num_classes <= 2:
            raise ValueError("Two classes or less cannot be trained"
                             "with softmax (multinomial).")

        if solves_classification and not solves_multiclass:
            self._num_classes_dim = self._num_classes - 1
        else:
            self._num_classes_dim = self._num_classes

        if self.fit_intercept:
            coef_size = (self.n_cols + 1, self._num_classes_dim)
        else:
            coef_size = (self.n_cols, self._num_classes_dim)

        if self.coef_ is None or not self.warm_start:
            self.solver_model._coef_ = CumlArray.zeros(
                coef_size, dtype=self.dtype, order='C')

    def fit(self, X, y, rank, n_ranks, n_samples, n_classes, convert_dtype=False) -> "LogisticRegressionMG":
        assert (n_classes == 2)

        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle() 

        X_m, n_rows, self.n_cols, self.dtype = input_to_cuml_array(
            X, check_dtype=[np.float32, np.float64], order = 'K'
        ) 

        y_m, label_rows, _, _ = input_to_cuml_array(
            y, check_dtype=self.dtype,
            convert_to_dtype=(self.dtype if convert_dtype else None),
            check_rows=n_rows, check_cols=1
        )

        cdef uintptr_t y_ptr = y_m.ptr
        cdef float objective32
        cdef int num_iters

        self._num_classes = n_classes
        self.prepare_for_fit(n_classes)

        cdef qn_params qnpams = self.qnparams.params
        cdef uintptr_t coef_ptr = self.coef_.ptr

        if self.dtype == np.float32:
            qnFit(
                handle_[0],
                qnpams,
                <float*><uintptr_t> X_m.ptr,
                <bool> __is_col_major(X_m),
                <float*> y_ptr,
                <int> n_rows,
                <int> self.n_cols,
                <int> self._num_classes,
                <float*> coef_ptr,
                <float*> &objective32,
                <int*> &num_iters,
                <int> n_samples,
                <int> rank,
                <int> n_ranks
            )

        self.solver_model._calc_intercept()

        self.handle.sync()
        del X_m
        del y_m
        return self
#
# Copyright (c) 2023, NVIDIA CORPORATION.
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

import cuml.internals
from cuml.internals.array import CumlArray
from cuml.linear_model.base_mg import MGFitMixin
from cuml.linear_model import LogisticRegression
from cuml.solvers.qn import QNParams
from cython.operator cimport dereference as deref

from pylibraft.common.handle cimport handle_t
from cuml.common.opg_data_utils_mg cimport *

# the cdef was copied from cuml.linear_model.qn
cdef extern from "cuml/linear_model/glm.hpp" namespace "ML::GLM" nogil:

    # TODO: Use single-GPU version qn_loss_type and qn_params https://github.com/rapidsai/cuml/issues/5502
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

    cdef void qnFit(
        handle_t& handle,
        vector[floatData_t *] input_data,
        PartDescriptor &input_desc,
        vector[floatData_t *] labels,
        float *coef,
        const qn_params& pams,
        bool X_col_major,
        int n_classes,
        float *f,
        int *num_iters) except +

    cdef vector[float] getUniquelabelsMG(
        const handle_t& handle,
        PartDescriptor &input_desc,
        vector[floatData_t*] labels) except+


class LogisticRegressionMG(MGFitMixin, LogisticRegression):

    def __init__(self, **kwargs):
        super(LogisticRegressionMG, self).__init__(**kwargs)
        if self.penalty != "l2" and self.penalty != "none":
            assert False, "Currently only support 'l2' and 'none' penalty"

    @property
    @cuml.internals.api_base_return_array_skipall
    def coef_(self):
        return self.solver_model.coef_

    @coef_.setter
    def coef_(self, value):
        # convert 1-D value to 2-D (to inherit MGFitMixin which sets self.coef_ to a 1-D array of length self.n_cols)
        if len(value.shape) == 1:
            new_shape=(1, value.shape[0])
            cp_array = value.to_output('array').reshape(new_shape)
            value, _, _, _ = input_to_cuml_array(cp_array, order='K')
            if (self.fit_intercept) and (self.solver_model.intercept_ is None):
                self.solver_model.intercept_ = CumlArray.zeros(shape=(1, 1), dtype = value.dtype)

        self.solver_model.coef_ = value

    def create_qnparams(self):
        return QNParams(
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

    def prepare_for_fit(self, n_classes):
        self.solver_model.qnparams = self.create_qnparams()

        # modified
        qnpams = self.solver_model.qnparams.params

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

    def fit(self, input_data, n_rows, n_cols, parts_rank_size, rank, convert_dtype=False):

        assert len(input_data) == 1, f"Currently support only one (X, y) pair in the list. Received {len(input_data)} pairs."
        self.is_col_major = False
        order = 'F' if self.is_col_major else 'C'
        super().fit(input_data, n_rows, n_cols, parts_rank_size, rank, order=order)

    @cuml.internals.api_base_return_any_skipall
    def _fit(self, X, y, coef_ptr, input_desc):
        cdef handle_t* handle_ = <handle_t*><size_t>self.handle.getHandle()
        cdef float objective32
        cdef int num_iters

        cdef vector[float] c_classes_
        c_classes_ = getUniquelabelsMG(
            handle_[0],
            deref(<PartDescriptor*><uintptr_t>input_desc),
            deref(<vector[floatData_t*]*><uintptr_t>y))
        self.classes_ = np.sort(list(c_classes_)).astype('float32')

        self._num_classes = len(self.classes_)
        self.loss = "sigmoid" if self._num_classes <= 2 else "softmax"
        self.prepare_for_fit(self._num_classes)
        cdef uintptr_t mat_coef_ptr = self.coef_.ptr

        cdef qn_params qnpams = self.solver_model.qnparams.params

        if self.dtype == np.float32:
            qnFit(
                handle_[0],
                deref(<vector[floatData_t*]*><uintptr_t>X),
                deref(<PartDescriptor*><uintptr_t>input_desc),
                deref(<vector[floatData_t*]*><uintptr_t>y),
                <float*>mat_coef_ptr,
                qnpams,
                self.is_col_major,
                self._num_classes,
                <float*> &objective32,
                <int*> &num_iters)
        else:
            assert False, "dtypes other than float32 are currently not supported yet. See issue: https://github.com/rapidsai/cuml/issues/5589"

        self.solver_model._calc_intercept()

        self.handle.sync()

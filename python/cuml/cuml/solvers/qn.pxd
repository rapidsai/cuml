# SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
from libcpp cimport bool

from cuml.internals.logger cimport level_enum


cdef extern from "cuml/linear_model/glm.hpp" namespace "ML::GLM" nogil:

    cdef enum Loss "ML::GLM::qn_loss_type":
        LOGISTIC "ML::GLM::QN_LOSS_LOGISTIC"
        SQUARED  "ML::GLM::QN_LOSS_SQUARED"
        SOFTMAX  "ML::GLM::QN_LOSS_SOFTMAX"
        SVC_L1   "ML::GLM::QN_LOSS_SVC_L1"
        SVC_L2   "ML::GLM::QN_LOSS_SVC_L2"
        SVR_L1   "ML::GLM::QN_LOSS_SVR_L1"
        SVR_L2   "ML::GLM::QN_LOSS_SVR_L2"
        ABS      "ML::GLM::QN_LOSS_ABS"

    cdef struct qn_params:
        Loss loss
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

cdef void init_qn_params(
    qn_params &params,
    int n_classes,
    loss,
    bool fit_intercept,
    double l1_strength,
    double l2_strength,
    int max_iter,
    double tol,
    delta,
    int linesearch_max_iter,
    int lbfgs_memory,
    bool penalty_normalized,
    level_enum verbose,
)

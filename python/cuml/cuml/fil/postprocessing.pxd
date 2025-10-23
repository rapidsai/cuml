#
# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
#
cdef extern from "cuml/fil/postproc_ops.hpp" namespace "ML::fil" nogil:
    cdef enum row_op:
        row_disable "ML::fil::row_op::disable",
        softmax "ML::fil::row_op::softmax",
        max_index "ML::fil::row_op::max_index"
    cdef enum element_op:
        elem_disable "ML::fil::element_op::disable",
        signed_square "ML::fil::element_op::signed_square",
        hinge "ML::fil::element_op::hinge",
        sigmoid "ML::fil::element_op::sigmoid",
        exponential "ML::fil::element_op::exponential",
        logarithm_one_plus_exp "ML::fil::element_op::logarithm_one_plus_exp"
